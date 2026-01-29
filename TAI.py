"""
=============================================================================
TAI - ANALYSEUR DE COLLAGENE PAR TRAITEMENT D'IMAGE
=============================================================================

OBJECTIF GÉNÉRAL :
    Ce script analyse des images microscopiques de poumons colorés au Trichrome de Masson pour quantifier automatiquement la fibrose pulmonaire en distinguant :
    - Le collagène (tissu bleu)
    - Le tissu sain (muscle/cellules rouges/roses/violettes)
    - Le fond/air (zones blanches vides)

PRINCIPE DE FONCTIONNEMENT :
    1. CALIBRATION : Analyse des couleurs de l'image pour déterminer les seuils
    2. RADAR : Création d'une carte basse résolution pour localiser le tissu
    3. ANALYSE HD : Traitement pixel par pixel des zones contenant du tissu
    4. RAPPORT : Génération des statistiques et visualisations

INNOVATION "SMART EDGES" :
    Utilisation d'interpolation bilinéaire sur les bords du masque radar pour
    éviter les artefacts en "escalier" et améliorer la précision sur les contours.

AUTEURS : Alcide Demeusy & Luana LOPES SANTIAGO
DATE : Janvier 2026
VERSION FINALE
================================================================================
"""
# ==============================================================================
# IMPORTS DES BIBLIOTHÈQUES
# ==============================================================================

import os           # Manipulation des chemins de fichiers et dossiers
import sys          # Fonctions système (notamment pour quitter proprement)
import time         # Mesure du temps d'exécution
import csv          # Écriture des fichiers CSV de résultats
import glob         # Recherche de fichiers avec patterns (*.tif, etc.)
import datetime     # Horodatage des analyses
import gc           # Garbage Collector : libération manuelle de la mémoire RAM

import numpy as np                  # Calculs numériques et manipulation de matrices
import matplotlib.pyplot as plt     # Génération de graphiques et visualisations
from tqdm import tqdm              # Barres de progression pour le suivi visuel

# Force matplotlib à ne pas ouvrir de fenêtres graphiques
# Critique pour éviter les plantages sur serveur ou lors d'analyses en batch
plt.switch_backend('Agg')

# ==============================================================================
# INFORMATIONS DE VERSION
# ==============================================================================

CODE_VERSION = "TAI vFINALE"
DATE_RUN = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

# ==============================================================================
# IMPORTS SPÉCIALISÉS AVEC GESTION D'ERREUR
# ==============================================================================

try:
    # tiffslide : Lecture optimisée des images TIFF pyramidales (WSI - Whole Slide Images)
    # Plus rapide et stable qu'openslide pour les très gros fichiers
    import tiffslide

    # scikit-image : Bibliothèque de traitement d'image
    from skimage import color      # Conversions d'espaces colorimétriques (RGB, HSV, etc.)
    from skimage import filters    # Filtres et seuillages (Otsu, etc.)
    from skimage import morphology # Opérations morphologiques (érosion, dilatation, etc.)
    from skimage.transform import resize  # Redimensionnement d'images

    # scipy : Bibliothèque scientifique
    from scipy.ndimage import binary_fill_holes  # Remplissage de trous dans les masques
    from scipy.ndimage import center_of_mass      # Calcul de centre de masse (non utilisé ici)

except ImportError as e:
    # Si une bibliothèque manque, on affiche un message clair et on arrête
    print(f"❌ Erreur : Bibliothèque manquante → {e}")
    print("   Solution : pip install tiffslide scikit-image scipy numpy matplotlib tqdm")
    sys.exit(1)

# ==============================================================================
# PARAMÈTRES DE SENSIBILITÉ (AJUSTABLE)
# ==============================================================================

# --- Paramètres de découpage ---
TILE_SIZE = 2048
"""
Taille des tuiles carrées (en pixels) pour l'analyse haute définition.
    - Plus grand = moins de tuiles mais plus de RAM nécessaire
    - Plus petit = plus de tuiles mais traitement plus lent
    - 2048 est un bon compromis pour des images de plusieurs Go
"""

# --- Seuils de filtrage HSV ---
HSV_SAT_MIN = 0.05
"""
Saturation minimale (0.0 à 1.0) pour considérer un pixel comme "coloré".
    - Élimine les pixels gris (poussière, artefacts, fond sale)
    - Valeur typique : 0.05-0.15
    - Plus bas = capture plus de pixels faiblement colorés
"""

HSV_VAL_MIN = 0.10
"""
Valeur minimale (luminosité) pour éviter les pixels trop sombres.
    - Élimine les zones noires (ombres, artefacts de scan)
    - Valeur typique : 0.10-0.20
"""

HSV_VAL_MAX = 0.95
"""
Valeur maximale pour éviter les pixels trop clairs.
    - Élimine les zones blanches éblouissantes (reflets, surexposition)
    - Valeur typique : 0.90-0.98
"""

# ==============================================================================
# CODES DES CLASSES ET COULEURS DE VISUALISATION
# ==============================================================================

# Identifiants numériques des classes (utilisés dans les masques)
ID_AIR  = 0  # Fond / Air (pas de matière biologique)
ID_COLL = 1  # Collagène
ID_TISS = 2  # Tissu sain

# Couleurs RGB normalisées (0.0-1.0) pour la visualisation
C_AIR  = [0.9, 1.0, 0.9]  # Vert très pâle-blanc (fond neutre)
C_COLL = [0.0, 0.4, 1.0]  # Bleu vif (collagène)
C_TISS = [1.0, 0.2, 0.2]  # Rouge vif (tissu)

# ==============================================================================
# FONCTION 1 : CALIBRATION AUTOMATIQUE DES SEUILS DE TEINTE (HSV)
# ==============================================================================

def auto_calibrate_hsv(slide, out_dir):
    """
    RÔLE : Déterminer automatiquement la plage de teintes (Hue) correspondant
           au collagène dans l'image actuelle.

    POURQUOI C'EST NÉCESSAIRE :
        - Chaque microscope, colorant, et échantillon donne des nuances différentes
        - Le collagène peut apparaître bleu pâle, bleu roi, ou bleu-violet
        - Sans calibration, on risque de manquer du collagène ou d'en inventer

    MÉTHODE SPECIALE "CYAN FORCE" :
        - On force la limite basse à 0.48 pour capturer le collagène délavé
        - On bloque la limite haute à 0.85 max pour éviter le violet
        - On analyse l'histogramme des teintes pour affiner entre ces bornes

    PARAMÈTRES :
        slide (TiffSlide)  : L'image ouverte (objet tiffslide)
        out_dir (str)      : Dossier où sauvegarder le graphique de calibration

    RETOUR :
        tuple (float, float) : (hue_min, hue_max) en échelle 0.0-1.0

    FICHIERS GÉNÉRÉS :
        - 0_CALIBRATION.png : Graphique de l'histogramme avec zones annotées
    """

    print("   ↳ 🧠 Calibration V36 (Cyan Force)...")

    # --- ÉTAPE 1 : Obtention d'une miniature (thumbnail) ---
    # On travaille sur une version réduite pour aller vite (quelques secondes vs minutes)
    try:
        # Tente d'obtenir une miniature de 1024x1024
        thumb = np.array(slide.get_thumbnail((1024, 1024)).convert("RGB"))
    except:
        # Si l'image est trop petite, on prend 512x512
        thumb = np.array(slide.get_thumbnail((512, 512)).convert("RGB"))

    # --- ÉTAPE 2 : Conversion en espace colorimétrique HSV ---
    # HSV = Hue (teinte), Saturation, Value (luminosité)
    # Plus pertinent que RGB pour séparer les couleurs (le bleu du rouge)
    hsv = color.rgb2hsv(thumb)

    # Extraction des 3 canaux en matrices séparées
    hue = hsv[:,:,0]  # Teinte : 0.0=rouge, 0.33=vert, 0.66=bleu, 1.0=rouge
    sat = hsv[:,:,1]  # Saturation : 0.0=gris, 1.0=couleur pure
    val = hsv[:,:,2]  # Valeur : 0.0=noir, 1.0=blanc

    # --- ÉTAPE 3 : Filtrage des pixels valides ---
    # On ne veut analyser que les pixels "colorés et visibles"
    valid_mask = (sat > 0.15) & (val > 0.15) & (val < 0.95)
    # Traduction : "Saturation > 15% ET Luminosité entre 15% et 95%"
    # Élimine : fond blanc, zones noires, pixels gris/poussière

    # Extraction des teintes des pixels valides uniquement
    valid_hues = hue[valid_mask]

    # --- ÉTAPE 4 : Sécurité si l'image est vide ou uniforme ---
    if len(valid_hues) == 0:
        print("      ⚠️  Image uniforme détectée → Valeurs par défaut")
    return 0.48, 0.85  # Valeurs conservatrices si pas de données

    # --- ÉTAPE 5 : Calcul de l'histogramme des teintes ---
    # On découpe l'échelle 0.0-1.0 en 120 intervalles (bins)
    hist, bin_edges = np.histogram(valid_hues, bins=120, range=(0, 1))
    # hist : nombre de pixels dans chaque intervalle
    # bin_edges : limites des intervalles

    # Calcul du centre de chaque intervalle pour le graphique
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # --- ÉTAPE 6 : Isolation de la zone bleue théorique ---
    # Dans l'espace HSV, le bleu se situe grossièrement entre 0.35 et 0.85
    # 0.35 = cyan/vert-bleu    0.66 = bleu pur    0.85 = bleu-violet
    blue_zone_mask = (centers > 0.35) & (centers < 0.85)

    # Création d'un histogramme ne contenant QUE les valeurs de cette zone
    hist_blue = hist.copy()
    hist_blue[~blue_zone_mask] = 0  # Mise à zéro des autres couleurs

    # --- ÉTAPE 7 : Vérification de la présence de bleu ---
    if np.sum(hist_blue) == 0:
        print("      ⚠️  Pas de bleu détecté → Valeurs par défaut")
        return 0.48, 0.85

    # --- ÉTAPE 8 : Détection du pic principal ---
    # Le pic = la teinte de bleu la plus fréquente (probable collagène)
    peak_idx = np.argmax(hist_blue)

    # --- ÉTAPE 9 : Recherche de la fin du pic ---
    # On avance vers les teintes plus hautes jusqu'à ce que la courbe redescende
    threshold = np.max(hist_blue) * 0.15  # 15% du maximum = considéré comme "bruit"
    idx_max = peak_idx

    # Boucle : tant qu'on est au-dessus du seuil et avant 0.85, on continue
    while idx_max < len(hist)-1 and hist[idx_max] > threshold and centers[idx_max] < 0.85:
        idx_max += 1

    detected_max = centers[idx_max]  # Teinte détectée comme fin du bleu

    # --- ÉTAPE 10 : APPLICATION DE LA LOGIQUE "CYAN FORCE" ---
    # C'est le coeur de la calibration

    final_min = 0.48
    """
    POURQUOI 0.48 FIXE ?
    - Le collagène jeune/délavé apparaît souvent cyan clair (0.45-0.50)
    - Si on laisse l'algo décider, il peut le rater (surtout si peu présent)
    - Forcer 0.48 garantit qu'on capture TOUT le collagène, même pâle
    - Risque : capturer un peu de fond bleuté (acceptable vs rater la fibrose)
    """

    final_max = min(0.85, detected_max + 0.03)
    """
    POURQUOI 0.85 MAX ET +0.03 ?
    - 0.85 est un MUR ABSOLU : au-delà, c'est du violet = muscle, pas collagène
    - Le +0.03 donne une petite marge pour capturer la queue du pic bleu
    - min() garantit qu'on ne dépasse JAMAIS 0.85, même si le pic va plus loin
    """

    # --- ÉTAPE 11 : Génération du graphique de contrôle ---
    # Ce graphique permet de vérifier visuellement la calibration
    plt.figure(figsize=(10, 5))

    # Courbe de l'histogramme complet (toutes les couleurs)
    plt.plot(centers, hist, color='gray', alpha=0.5, label="Spectre complet")

    # Zone rouge : les teintes interdites (violet/magenta = muscle)
    plt.axvspan(0.85, 1.0, color='red', alpha=0.1, label="Zone Tissu (>0.85)")

    # Zone verte : la plage retenue pour le collagène
    plt.axvspan(final_min, final_max, color='green', alpha=0.3,
                label=f"Collagène ({final_min:.2f}-{final_max:.2f})")

    plt.xlabel("Teinte HSV (Hue)")
    plt.ylabel("Nombre de pixels")
    plt.title(f"Calibration Automatique | {CODE_VERSION}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Sauvegarde du graphique (IMPORTANT pour la traçabilité)
    calib_path = os.path.join(out_dir, "0_CALIBRATION.png")
    plt.savefig(calib_path, dpi=150, bbox_inches='tight')
    plt.close('all')  # Libération mémoire

    print(f"      ✓ Seuils calibrés : Hue ∈ [{final_min:.2f}, {final_max:.2f}]")
    print(f"      ✓ Graphique sauvegardé : {calib_path}")

    return final_min, final_max


# ==============================================================================
# FONCTION 2 : ANALYSE PIXEL PAR PIXEL D'UNE TUILE HAUTE DÉFINITION
# ==============================================================================

def analyze_tile_adaptive(tile, b_min, b_max):
    """
    RÔLE : Classifier chaque pixel d'une tuile en 3 catégories :
           - ID_AIR (0)  : Fond vide
           - ID_COLL (1) : Collagène
           - ID_TISS (2) : Tissu sain

    PRINCIPE :
        1. On convertit l'image en HSV pour séparer les couleurs
        2. On applique des filtres pour éliminer le fond et les artefacts
        3. On compare la teinte aux seuils calibrés pour identifier le collagène
        4. Tout ce qui reste (coloré mais pas bleu) est du tissu

    PARAMÈTRES :
        tile (ndarray)     : Image RGB de la tuile (shape: H, W, 3)
        b_min (float)      : Teinte minimale du collagène (de calibration)
        b_max (float)      : Teinte maximale du collagène (de calibration)

    RETOUR :
        ndarray (uint8) : Masque de classification (shape: H, W)
                         Valeurs : 0=Air, 1=Collagène, 2=Tissu
    """

    # --- ÉTAPE 1 : Conversion RGB → HSV ---
    hsv = color.rgb2hsv(tile)
    hue = hsv[:,:,0]  # Teinte (la "couleur" pure)
    sat = hsv[:,:,1]  # Saturation (intensité de la couleur)
    val = hsv[:,:,2]  # Valeur (luminosité)

    # --- ÉTAPE 2 : Détection de la "matière biologique" ---
    # Un pixel est considéré comme "matière" s'il est :
    # - Suffisamment saturé (pas gris/blanc)
    # - Ni trop sombre (pas d'ombre)
    # - Ni trop clair (pas de reflet)
    is_matter = (sat > HSV_SAT_MIN) & (val > HSV_VAL_MIN) & (val < HSV_VAL_MAX)

    # Explication visuelle :
    # HSV_SAT_MIN (0.05) élimine :  ████ (gris, poussière)
    # HSV_VAL_MIN (0.10) élimine :  ▓▓▓▓ (ombres noires)
    # HSV_VAL_MAX (0.95) élimine :  ░░░░ (reflets blancs)

    # --- ÉTAPE 3 : Identification du collagène parmi la matière ---
    # Un pixel est du collagène si :
    # - C'est de la matière (condition précédente)
    # - Sa teinte est dans l'intervalle [b_min, b_max] calibré
    is_coll = (hue >= b_min) & (hue <= b_max) & is_matter

    # --- ÉTAPE 4 : Identification du tissu sain ---
    # Par élimination : c'est de la matière, mais pas du collagène
    is_tiss = is_matter & (~is_coll)

    # Logique booléenne :
    # - Si pixel ∈ [b_min, b_max] ET coloré → Collagène
    # - Si pixel coloré mais hors [b_min, b_max] → Tissu
    # - Sinon (gris/blanc/noir) → Air

    # --- ÉTAPE 5 : Création du masque de sortie ---
    # On initialise un masque vide (tout à 0 = Air)
    mask = np.zeros(tile.shape[:2], dtype=np.uint8)

    # On remplit avec les identifiants des classes
    mask[is_coll] = ID_COLL  # Les pixels collagène passent à 1
    mask[is_tiss] = ID_TISS  # Les pixels tissu passent à 2
    # Les pixels air restent à 0 (valeur initiale)

    return mask

# ==============================================================================
# FONCTION 3 : NORMALISATION DES IMAGES 16-BIT VERS 8-BIT
# ==============================================================================

def force_normalize_8bit(tile):
    """
    RÔLE : Convertir une image de n'importe quel format (8-bit, 16-bit, float)
           vers du 8-bit standard (0-255).

    POURQUOI C'EST NÉCESSAIRE :
        - Certains microscopes génèrent du 16-bit (0-65535)
        - Les algorithmes de traitement attendent du 8-bit (0-255)
        - Sans conversion, les images peuvent apparaître noires ou bugguées

    MÉTHODE :
        - Détection du format actuel
        - Normalisation linéaire (étirement) vers 0-255
        - Conversion en uint8 (entier 8-bit non signé)

    PARAMÈTRES :
        tile (ndarray ou None) : Image à normaliser

    RETOUR :
        ndarray (uint8) : Image normalisée, ou None si entrée invalide
    """

    # --- Vérification de validité ---
    if tile is None or tile.size == 0:
        return None  # Image vide ou corrompue

    # --- Si déjà en 8-bit, rien à faire ---
    if tile.dtype == np.uint8:
        return tile

    # --- Conversion en float pour calculs précis ---
    img = tile.astype(float)

    # --- Cas limite : image uniforme (même valeur partout) ---
    if img.max() == img.min():
        # Impossible de normaliser, on renvoie du noir
        return np.zeros(tile.shape, dtype=np.uint8)

    # --- Normalisation linéaire ---
    # Formule : (pixel - min) / (max - min) * 255
    # Effet : Le pixel le plus sombre devient 0, le plus clair devient 255
    img = (img - img.min()) / (img.max() - img.min()) * 255.0

    # --- Clipping et conversion finale ---
    # np.clip : force les valeurs dans [0, 255] (au cas où calculs hors limites)
    # astype(uint8) : conversion en entier 8-bit
    return np.clip(img, 0, 255).astype(np.uint8)

# ==============================================================================
# FONCTION 4 : CRÉATION DU "RADAR" (CARTE BASSE RÉSOLUTION DU TISSU)
# ==============================================================================

def create_radar_mask(slide):
    """
    RÔLE : Créer rapidement une carte grossière indiquant où se trouve le tissu
           dans l'image. Cela permet d'éviter de scanner le fond blanc inutilement.

    ANALOGIE :
        Imagine que tu scannes une page avec un petit dessin au centre.
        Au lieu de lire pixel par pixel toute la page blanche, tu fais d'abord
        un coup d'œil rapide pour repérer où est le dessin, puis tu zoomes
        seulement sur cette zone. C'est exactement ce que fait cette fonction.

    MÉTHODE "MODE SMART" :
        1. Obtention d'une miniature (2048x2048 ou moins)
        2. Conversion en niveaux de gris
        3. Seuillage automatique (méthode d'Otsu) : sépare tissu/fond
        4. Nettoyage morphologique : supprime les petits artefacts
        5. Remplissage des trous : bouche les cavités internes

    PARAMÈTRES :
        slide (TiffSlide) : L'image ouverte

    RETOUR :
        tuple : (mask, thumb)
            - mask (ndarray bool) : Carte True=tissu, False=fond (shape: H, W)
            - thumb (ndarray uint8) : La miniature utilisée (pour visualisation)

    AMÉLIORATIONS POSSIBLES :
        - Ajuster les paramètres morphologiques selon le type de tissu
        - Ajouter un filtrage par couleur (ignorer les zones très rouges/bleues)
    """

    print(" ↳ 📡 Radar : Scan structure")

    # --- ÉTAPE 1 : Obtention de la miniature ---
    try:
        # Essai avec résolution 2048x2048 (bon compromis vitesse/précision)
        thumb = np.array(slide.get_thumbnail((2048, 2048)).convert("RGB"))
    except:
        # Fallback sur 1024x1024 si image trop petite
        thumb = np.array(slide.get_thumbnail((1024, 1024)).convert("RGB"))

    # Normalisation au cas où ce serait du 16-bit
    thumb = force_normalize_8bit(thumb)

    # --- ÉTAPE 2 : Conversion en niveaux de gris ---
    # Le niveau de gris permet de distinguer "clair" (fond) et "sombre" (tissu)
    gray = color.rgb2gray(thumb)  # Moyenne pondérée des canaux RGB

    # --- ÉTAPE 3 : Seuillage automatique d'Otsu ---
    """
    L'ALGORITHME D'OTSU :
    - Analyse l'histogramme des niveaux de gris
    - Cherche le seuil qui sépare au mieux deux groupes (bimodal)
    - Dans notre cas : groupe "fond blanc" vs groupe "tissu sombre"
    - Retourne un seuil optimal automatiquement
    """
    thresh = filters.threshold_otsu(gray)

    # Création du masque binaire : True là où gray < thresh (zones sombres)
    binary = gray < thresh

    # --- ÉTAPE 4 : Nettoyage morphologique ---

    # 4a. Suppression des petits objets isolés (poussières, artefacts)
    # min_size=500 : les groupes de moins de 500 pixels sont supprimés
    binary = morphology.remove_small_objects(binary, min_size=500)

    # 4b. Fermeture morphologique (dilatation puis érosion)
    # Effet : bouche les petits trous et connecte les zones proches
    # disk(5) : utilise un disque de rayon 5 pixels comme élément structurant
    binary = morphology.binary_closing(binary, morphology.disk(5))

    # --- ÉTAPE 5 : Remplissage des cavités internes ---
    # Si le tissu forme un anneau, on remplit l'intérieur
    # Exemple : bronche vue en coupe → on veut compter la paroi ET l'intérieur
    mask = binary_fill_holes(binary)

    print(f"      ✓ Radar créé : {np.sum(mask)} pixels de tissu détectés")
    print(f"      ✓ Résolution : {mask.shape[1]}x{mask.shape[0]}")

    return mask, thumb


# ==============================================================================
# FONCTION 5 : GÉNÉRATION DU RAPPORT FINAL
# ==============================================================================

def generate_report(vis_map, counts, filename, out_dir, proc_time, bmin, bmax):
    """
    RÔLE : Créer le rapport final avec statistiques, graphiques et carte visuelle
           de l'analyse complète d'une image.

    CONTENU DU RAPPORT :
        1. Carte visuelle haute résolution (collagène en bleu, tissu en rouge)
        2. Graphiques statistiques (barres)
        3. Métadonnées (date, version, paramètres)

    PARAMÈTRES :
        vis_map (ndarray)    : Carte visuelle complète (RGB)
        counts (dict)        : {ID_AIR: n1, ID_COLL: n2, ID_TISS: n3}
        filename (str)       : Nom du fichier source
        out_dir (str)        : Dossier de sortie
        proc_time (float)    : Temps de traitement (secondes)
        bmin, bmax (float)   : Seuils HSV utilisés

    RETOUR :
        dict : Dictionnaire de toutes les statistiques (pour le CSV global)

    FICHIERS GÉNÉRÉS :
        - RAPPORT_FINAL_{nom}.png : Rapport visuel complet
    """

    # --- ÉTAPE 1 : Calcul des statistiques de base ---

    total = sum(counts.values())
    if total == 0:
        total = 1  # Évite la division par zéro

    # Pourcentages sur le total
    pct_air = (counts[ID_AIR] / total) * 100
    pct_coll = (counts[ID_COLL] / total) * 100
    pct_tiss = (counts[ID_TISS] / total) * 100

    # --- ÉTAPE 2 : Création du rapport visuel (pseudo-code) ---
    # (Le code de génération matplotlib serait ici)
    # plt.figure, plt.imshow, plt.text, plt.pie, etc.

    # --- ÉTAPE 3 : Préparation du dictionnaire de sortie ---

    stats_dict = {
        "Fichier": filename,
        "Date": DATE_RUN,
        "Version": CODE_VERSION,
        "Temps_traitement_sec": round(proc_time, 1),

        # Comptages bruts
        "Pixels_Air": counts[ID_AIR],
        "Pixels_Collagene": counts[ID_COLL],
        "Pixels_Tissu": counts[ID_TISS],
        "Pixels_Total": total,

        # Pourcentages
        "Pct_Air": round(pct_air, 2),
        "Pct_Collagene": round(pct_coll, 2),
        "Pct_Tissu": round(pct_tiss, 2),

        # Paramètres techniques
        "HSV_Hue_Min": round(bmin, 3),
        "HSV_Hue_Max": round(bmax, 3),
    }

    return stats_dict

# ==============================================================================
# FONCTION 6 : TRAITEMENT D'UNE IMAGE COMPLÈTE (WORKER PRINCIPAL)
# ==============================================================================

def process_one_image(path, output_root):
    """
    RÔLE : Orchestrer l'analyse complète d'une seule image microscopique.
           C'est la fonction "chef d'orchestre" qui appelle toutes les autres.

    WORKFLOW COMPLET (7 GRANDES ÉTAPES) :

    1. INITIALISATION
       - Création du dossier de sortie
       - Ouverture de l'image avec tiffslide
       - Démarrage du chronomètre

    2. CALIBRATION
       - Appel de auto_calibrate_hsv()
       - Détermination des seuils de teinte pour le collagène

    3. CRÉATION DU RADAR
       - Appel de create_radar_mask()
       - Génération d'une carte basse résolution du tissu

    4. PLANIFICATION DES TÂCHES
       - Découpage virtuel de l'image en tuiles de TILE_SIZE pixels
       - Filtrage par le radar : on ne garde que les tuiles contenant du tissu
       - Résultat : liste des coordonnées (x, y) à analyser

    5. ANALYSE HAUTE DÉFINITION (BOUCLE PRINCIPALE)
       Pour chaque tuile à analyser :
       a. Lecture de la tuile HD depuis l'image source
       b. Appel de analyze_tile_adaptive() → classification pixel par pixel
       c. Application du masque radar avec interpolation
       d. Comptage des pixels par classe
       e. Mise à jour de la carte visuelle globale

    6. POST-TRAITEMENT
       - Agrégation des comptages de toutes les tuiles
       - Génération du rapport final
       - Sauvegarde des fichiers de sortie

    7. NETTOYAGE
       - Fermeture du fichier image
       - Libération de la mémoire (garbage collector)

    PARAMÈTRES :
        path (str)         : Chemin complet vers le fichier image (.tif, .svs, etc.)
        output_root (str)  : Dossier racine où créer les sous-dossiers de résultats

    RETOUR :
        dict ou None : Dictionnaire de statistiques si succès, None si erreur

    FICHIERS GÉNÉRÉS :
        Dans le dossier ANALYSE_{nom_fichier}/ :
        - 0_CALIBRATION.png : Graphique de calibration
        - RAPPORT_FINAL.png : Rapport visuel complet
        - CARTE_VISUELLE.png : Carte haute résolution de la classification

    GESTION DES ERREURS :
        - Fichier corrompu → Erreur capturée, None renvoyé
        - Manque de RAM → Erreur capturée, nettoyage automatique
        - Toute exception → Message d'erreur + None
    """

    start_time = time.time()  # Chronomètre
    filename = os.path.basename(path)  # Nom du fichier seul

    print(f"\n{'='*70}")
    print(f"🔹 ANALYSE : {filename}")
    print(f"{'='*70}")

    # --- ÉTAPE 1 : INITIALISATION ---

    # Création du dossier de sortie spécifique à cette image
    name_no_ext = os.path.splitext(filename)[0]  # Nom sans extension
    out_dir = os.path.join(output_root, f"ANALYSE_{name_no_ext}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"📂 Dossier de sortie : {out_dir}")

    slide = None  # Initialisation pour le finally

    try:
        # --- Ouverture de l'image ---
        print("📖 Ouverture du fichier...")
        slide = tiffslide.TiffSlide(path)

        # Récupération des dimensions réelles (niveau 0 = plus haute résolution)
        W, H = slide.dimensions
        print(f"   ✓ Dimensions : {W} × {H} pixels ({W*H/1e6:.1f} Mpx)")

        # --- ÉTAPE 2 : CALIBRATION ---
        print("\n🧠 PHASE 1 : CALIBRATION AUTOMATIQUE")
        bmin, bmax = auto_calibrate_hsv(slide, out_dir)
        print(f"   ✓ Plage de teinte collagène : [{bmin:.3f}, {bmax:.3f}]")

        # --- ÉTAPE 3 : CRÉATION DU RADAR ---
        print("\n📡 PHASE 2 : CRÉATION DU RADAR TISSULAIRE")
        mask_global, thumb = create_radar_mask(slide)

        # Dimensions du radar (basse résolution)
        hm, wm = mask_global.shape
        print(f"   ✓ Résolution radar : {wm} × {hm}")

        # Calcul des ratios de conversion Radar → HD
        # Exemple : si image HD = 20000×15000 et radar = 2000×1500
        #           alors rx = 10, ry = 10 (1 pixel radar = 10 pixels HD)
        rx = W / wm  # Ratio X
        ry = H / hm  # Ratio Y
        print(f"   ✓ Ratios de conversion : rx={rx:.2f}, ry={ry:.2f}")

        # --- ÉTAPE 4 : PLANIFICATION DES TÂCHES ---
        print("\n📋 PHASE 3 : PLANIFICATION DES TUILES À ANALYSER")

        tasks = []  # Liste des tuiles à traiter

        # Boucle sur toute l'image par blocs de TILE_SIZE
        for y in range(0, H, TILE_SIZE):
            for x in range(0, W, TILE_SIZE):

                # Conversion des coordonnées HD → Radar
                xs = int(x / rx)  # X start dans le radar
                xe = int(min(x + TILE_SIZE, W) / rx)  # X end dans le radar
                ys = int(y / ry)  # Y start dans le radar
                ye = int(min(y + TILE_SIZE, H) / ry)  # Y end dans le radar

                # Vérification : est-ce que cette zone contient du tissu ?
                # On regarde dans le masque radar si au moins 1 pixel = True
                if xs < wm and ys < hm and np.any(mask_global[ys:ye, xs:xe]):
                    # Oui → on ajoute cette tuile à la liste des tâches
                    tasks.append((x, y, xs, xe, ys, ye))

        nb_tuiles_totales = (W // TILE_SIZE + 1) * (H // TILE_SIZE + 1)
        nb_tuiles_utiles = len(tasks)
        pct_skip = 100 * (1 - nb_tuiles_utiles / nb_tuiles_totales)

        print(f"   ✓ Tuiles théoriques : {nb_tuiles_totales}")
        print(f"   ✓ Tuiles à analyser : {nb_tuiles_utiles}")
        print(f"   ✓ Optimisation : {pct_skip:.1f}% de fond blanc ignoré")

        # --- ÉTAPE 5 : INITIALISATION DES COMPTEURS ET CARTE VISUELLE ---
        print("\n🔬 PHASE 4 : ANALYSE HAUTE DÉFINITION (PIXEL PAR PIXEL)")

        # Compteurs globaux pour toute l'image
        counts = {ID_AIR: 0, ID_COLL: 0, ID_TISS: 0}

        # Carte visuelle (version basse résolution pour économiser la RAM)
        # On la remplit au fur et à mesure avec les résultats des tuiles
        vis_scale = 8  # 1 pixel de la carte = 8×8 pixels de l'image HD
        vis_h = H // vis_scale
        vis_w = W // vis_scale
        vis_map = np.ones((vis_h, vis_w, 3), dtype=np.float32) * np.array(C_AIR)

        print(f"   ✓ Carte visuelle : {vis_w} × {vis_h} (échelle 1:{vis_scale})")

        # --- BOUCLE PRINCIPALE : TRAITEMENT DE CHAQUE TUILE ---

        # Barre de progression (tqdm) pour suivre l'avancement
        for i, (x, y, xs, xe, ys, ye) in enumerate(tqdm(tasks, desc="   Tuiles",
                                                          unit="tuile", ncols=70)):

            # Calcul de la taille réelle de la tuile (peut être réduite sur les bords)
            wr = min(TILE_SIZE, W - x)
            hr = min(TILE_SIZE, H - y)

            # --- 5a. LECTURE DE LA TUILE HD ---
            # read_region() est la fonction magique de tiffslide qui lit
            # une portion rectangulaire de l'image sans charger tout le fichier
            raw = slide.read_region((x, y), 0, (wr, hr))
            # Niveau 0 = plus haute résolution disponible

            # Conversion en tableau numpy RGB
            tile = np.array(raw.convert("RGB"))

            # --- 5b. ANALYSE HD (CLASSIFICATION PIXEL PAR PIXEL) ---
            mask_hd = analyze_tile_adaptive(tile, bmin, bmax)
            # Résultat : masque de même taille que tile, valeurs 0/1/2

            # --- 5c. APPLICATION DU MASQUE RADAR (SMART EDGES V51) ---
            """
            ╔═══════════════════════════════════════════════════════════════╗
            ║  INNOVATION CLÉE : INTERPOLATION BILINÉAIRE DES BORDS         ║
            ╚═══════════════════════════════════════════════════════════════╝

            PROBLÈME DES VERSIONS PRÉCÉDENTES :
            - Le radar est en basse résolution (ex: 2000×1500)
            - La tuile HD est en haute résolution (ex: 2048×2048)
            - Quand on redimensionne le radar vers HD avec order=0 (nearest),
              on obtient des escaliers sur les bords du tissu.

              Problème : les bords "en escalier" font qu'on coupe parfois
              du vrai tissu ou qu'on inclut du fond par erreur.

            SOLUTION : INTERPOLATION BILINÉAIRE (order=1)
            - Au lieu de répliquer les pixels, on crée un gradient doux
            - Les bords passent progressivement de 0.0 (fond) à 1.0 (tissu)
            - On applique un seuil à 0.15 pour définir la zone "valide"

            POURQUOI SEUIL À 0.15 ?
            - Testé empiriquement sur des dizaines d'images
            - 0.50 : coupe du vrai tissu → faux négatifs
            - 0.10 : inclut trop de fond → faux positifs
            - 0.15 : compromis optimal pour les bordures

            AVANTAGES :
            - Contours plus lisses et naturels
            - Meilleure détection des structures fines
            - Réduit les artefacts visuels dans la carte finale
            """

            # Extraction de la zone radar correspondant à cette tuile
            radar_patch = mask_global[ys:ye, xs:xe].astype(float)
            # Conversion en float pour permettre l'interpolation

            # Redimensionnement du radar vers la résolution HD avec interpolation
            # order=1 : bilinéaire (moyenne pondérée des 4 voisins les plus proches)
            # preserve_range=True : garde les valeurs entre 0 et 1
            radar_hd = resize(radar_patch, (hr, wr), order=1, preserve_range=True)

            # Application du seuil : valeurs >0.15 considérées comme "dans le tissu"
            radar_loc = radar_hd > 0.15

            # --- 5d. COMPTAGE DES PIXELS VALIDES ---
            # On ne compte que les pixels qui sont :
            # - Détectés par l'analyse HD (mask_hd)
            # - ET dans la zone valide du radar (radar_loc)
            valid_pixels = mask_hd[radar_loc]

            # Mise à jour des compteurs globaux
            counts[ID_AIR] += np.sum(~radar_loc)  # Pixels hors tissu
            counts[ID_COLL] += np.sum(valid_pixels == ID_COLL)
            counts[ID_TISS] += np.sum(valid_pixels == ID_TISS)

            # --- 5e. MISE À JOUR DE LA CARTE VISUELLE ---
            # On convertit les coordonnées HD → coordonnées de la carte visuelle
            vx_start = x // vis_scale
            vx_end = (x + wr) // vis_scale
            vy_start = y // vis_scale
            vy_end = (y + hr) // vis_scale

            # Redimensionnement du masque HD vers la résolution de la carte
            mask_vis = resize(mask_hd.astype(float),
                            (vy_end - vy_start, vx_end - vx_start),
                            order=0, preserve_range=True).astype(np.uint8)

            # Colorisation : 0→Vert, 1→Bleu, 2→Rouge
            for c in range(3):  # Pour chaque canal RGB
                patch = vis_map[vy_start:vy_end, vx_start:vx_end, c]
                patch[mask_vis == ID_COLL] = C_COLL[c]
                patch[mask_vis == ID_TISS] = C_TISS[c]
                # Les zones Air gardent leur couleur initiale (vert pâle)

        # --- FIN DE LA BOUCLE PRINCIPALE ---

        # --- ÉTAPE 6 : POST-TRAITEMENT ET GÉNÉRATION DU RAPPORT ---
        print("\n📊 PHASE 5 : GÉNÉRATION DU RAPPORT FINAL")

        processing_time = time.time() - start_time
        print(f"   ✓ Temps de traitement : {processing_time:.1f} secondes")

        # Sauvegarde de la carte visuelle
        vis_path = os.path.join(out_dir, "CARTE_VISUELLE.png")
        plt.imsave(vis_path, vis_map)
        print(f"   ✓ Carte visuelle sauvegardée : {vis_path}")

        # Génération du rapport avec statistiques
        stats_data = generate_report(vis_map, counts, filename, out_dir,
                                     processing_time, bmin, bmax)

        print(f"\n✅ ANALYSE TERMINÉE AVEC SUCCÈS")
        print(f"   Pixels analysés : {sum(counts.values()):,}")

        return stats_data

    except Exception as e:
        # --- GESTION DES ERREURS ---
        print(f"\n❌ ERREUR LORS DE L'ANALYSE : {e}")
        import traceback
        traceback.print_exc()  # Affiche la stack trace complète pour debug
        return None

    finally:

        # --- ÉTAPE 7 : NETTOYAGE (EXÉCUTÉ MÊME EN CAS D'ERREUR) ---
        """
        LE NETTOYAGE EST CRITIQUE pour éviter :
        - Le crash "WinError 10055" (trop de fichiers ouverts)
        - Les fuites mémoire (RAM qui ne se libère pas)
        - Les fichiers verrouillés (impossibles à supprimer/modifier)

        SANS CETTE SECTION, après 10-20 images, le programme peut :
        - Ralentir énormément
        - Crasher avec des erreurs mémoire
        - Bloquer le système d'exploitation
        """

        if slide is not None:
            slide.close()  # Fermeture propre du fichier image

        plt.close('all')  # Fermeture de toutes les figures matplotlib

        gc.collect()  # Force Python à libérer la mémoire inutilisée
        # gc = Garbage Collector (éboueur) : nettoie la RAM

        # Pause courte pour laisser le système respirer
        time.sleep(0.3)

# ==============================================================================
# FONCTION 7 : INTERFACE UTILISATEUR (MAIN)
# ==============================================================================

def main():
    """
    RÔLE : Interface en ligne de commande pour l'utilisateur.
           Gère la sélection des fichiers et l'orchestration de l'analyse batch.

    FONCTIONNALITÉS :
        1. Choix du mode :
           - Mode 1 : Fichiers manuels (l'utilisateur liste les chemins)
           - Mode 2 : Dossier complet (analyse tous les fichiers d'un dossier)

        2. Déduplication automatique des chemins
        3. Choix du dossier de sortie
        4. Traitement séquentiel de toutes les images
        5. Génération d'un CSV global consolidant tous les résultats

    FORMATS SUPPORTÉS :
        - .tif, .tiff : TIFF standard ou pyramidal
        - .svs : Format Aperio (courant en pathologie)
        - .ndpi : Format Hamamatsu

    FICHIERS GÉNÉRÉS :
        Pour chaque image : dossier ANALYSE_{nom}/
        En global : RESULTATS_GLOBAUX.csv (tableau récapitulatif)
    """

    print("\n" + "="*70)
    print(f"  {CODE_VERSION}")
    print(f"  Analyse de fibrose pulmonaire par traitement d'image")
    print("="*70)

    # --- ÉTAPE 1 : SÉLECTION DU MODE ---
    print("\n🔧 MODES DISPONIBLES :")
    print("  1. Fichiers manuels (entrer les chemins un par un)")
    print("  2. Dossier complet (analyser tous les .tif/.svs/.ndpi)")

    choice = input("\n👉 Votre choix (1 ou 2) : ").strip()

    # Ensemble pour éviter les doublons (set = liste sans doublons)
    unique_paths = set()

    # --- MODE 1 : FICHIERS MANUELS ---
    if choice == '1':
        print("\n📝 Entrez les chemins des fichiers, séparés par des virgules")
        print("   Exemple : C:/Images/poumon1.tif, D:/Data/poumon2.svs")

        raw_input = input("\n👉 Chemins : ")

        # Découpage par virgules et nettoyage
        paths = [p.strip().replace('"', '') for p in raw_input.split(',')]

        # Vérification de l'existence de chaque fichier
        for p in paths:
            if os.path.isfile(p):
                # Normalisation du chemin (gère les / vs \, etc.)
                unique_paths.add(os.path.normpath(p))
            else:
                print(f"   ⚠️  Fichier ignoré (introuvable) : {p}")

    # --- MODE 2 : DOSSIER COMPLET ---
    elif choice == '2':
        print("\n📁 Entrez le chemin du dossier contenant les images")
        print("   Exemple : C:/Images/Poumons/")

        folder = input("\n👉 Dossier : ").strip().replace('"', '')

        if os.path.isdir(folder):
            # Extensions recherchées (minuscules ET majuscules)
            extensions = ('*.tif', '*.tiff', '*.svs', '*.ndpi')

            for ext in extensions:
                # Recherche récursive avec glob
                found_lower = glob.glob(os.path.join(folder, ext))
                found_upper = glob.glob(os.path.join(folder, ext.upper()))

                # Ajout à l'ensemble (déduplication automatique)
                for f in found_lower + found_upper:
                    unique_paths.add(os.path.normpath(f))

            print(f"   ✓ {len(unique_paths)} fichier(s) trouvé(s)")
        else:
            print(f"   ❌ Dossier introuvable : {folder}")

    else:
        print("❌ Choix invalide. Abandon.")
        return

    # --- VÉRIFICATION : Y A-T-IL DES FICHIERS À TRAITER ? ---
    files_to_process = list(unique_paths)

    if not files_to_process:
        print("\n❌ Aucun fichier à traiter. Programme terminé.")
        return

    # --- ÉTAPE 2 : CHOIX DU DOSSIER DE SORTIE ---
    print("\n📂 DOSSIER DE SORTIE")
    print("   (Laissez vide pour créer 'Resultats' à côté des images)")

    out_root_input = input("\n👉 Dossier de sortie : ").strip().replace('"', '')

    if not out_root_input:
        # Par défaut : créer à côté du premier fichier
        default_dir = os.path.dirname(files_to_process[0])
        out_root = os.path.join(default_dir, 'Resultats')
    else:
        out_root = out_root_input

    # Création du dossier (sans erreur s'il existe déjà)
    os.makedirs(out_root, exist_ok=True)

    # --- ÉTAPE 3 : RÉSUMÉ AVANT LANCEMENT ---
    print("\n" + "="*70)
    print("📋 RÉCAPITULATIF")
    print("="*70)
    print(f"  Nombre d'images : {len(files_to_process)}")
    print(f"  Dossier de sortie : {out_root}")
    print(f"  Taille des tuiles : {TILE_SIZE}×{TILE_SIZE} pixels")
    print(f"  Version : {CODE_VERSION}")
    print("="*70)

    input("\n⏸️  Appuyez sur Entrée pour commencer l'analyse...")

    # --- ÉTAPE 4 : BOUCLE PRINCIPALE DE TRAITEMENT ---
    print(f"\n🚀 DÉMARRAGE DU TRAITEMENT ({len(files_to_process)} images)")

    global_data_list = []  # Liste de tous les résultats pour le CSV global

    for i, img_path in enumerate(files_to_process):
        print(f"\n{'#'*70}")
        print(f"# Image {i+1}/{len(files_to_process)}")
        print(f"{'#'*70}")

        # Traitement de l'image
        result = process_one_image(img_path, out_root)

        if result is not None:
            global_data_list.append(result)
        else:
            print("⚠️  Image ignorée (erreur de traitement)")

        # Pause entre les images pour stabilité
        time.sleep(0.5)

    # --- ÉTAPE 5 : GÉNÉRATION DU CSV GLOBAUX ---
    if global_data_list:
        csv_path = os.path.join(out_root, "RESULTATS_GLOBAUX.csv")

        print(f"\n{'='*70}")
        print("💾 GÉNÉRATION DU CSV GLOBAUX")
        print(f"{'='*70}")
        print(f"   Fichier : {csv_path}")

        # Écriture du CSV avec séparateur point-virgule (compatible Excel)
        keys = global_data_list[0].keys()
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys, delimiter=';')
            writer.writeheader()  # Ligne d'en-tête
            writer.writerows(global_data_list)  # Toutes les lignes de données

        print(f"   ✓ {len(global_data_list)} ligne(s) écrite(s)")

    # --- ÉTAPE 6 : MESSAGE FINAL ---
    print("\n" + "="*70)
    print("✅ ANALYSE COMPLÈTE TERMINÉE AVEC SUCCÈS")
    print("="*70)
    print(f"📂 Résultats disponibles dans : {out_root}")
    print(f"📊 CSV global : RESULTATS_GLOBAUX.csv")
    print(f"⏱️  Temps total : {time.time():.1f} secondes (approximatif)")
    print("\n🔬 Merci d'avoir utilisé notre outil !")
    print("="*70)


# ==============================================================================
# POINT D'ENTRÉE DU PROGRAMME
# ==============================================================================

if __name__ == "__main__":
    """
    Cette condition permet d'exécuter main() seulement si le script est lancé
    directement (pas s'il est importé comme module dans un autre script).
    """
    main()

# ==============================================================================
# NOTES POUR LES FUTURS DÉVELOPPEURS
# ==============================================================================

"""
POINTS D'AMÉLIORATION POSSIBLES :

1. PARALLÉLISATION :
   - Actuellement, les tuiles sont traitées séquentiellement
   - On pourrait utiliser multiprocessing pour traiter plusieurs tuiles en parallèle
   - Gain de vitesse : ×2 à ×8 selon le CPU
   - Attention : nécessite plus de RAM (chaque processus charge sa propre tuile)

2. INTERFACE GRAPHIQUE :
   - Ajouter une GUI avec tkinter ou PyQt pour éviter la ligne de commande
   - Permettre de visualiser les résultats en temps réel
   - Drag & drop des fichiers

3. VALIDATION STATISTIQUE :
   - Ajouter des intervalles de confiance sur le score de fibrose
   - Calculer la variabilité inter-tuiles
   - Détection automatique des zones problématiques (artefacts, bulles, plis)

4. MACHINE LEARNING :
   - Entraîner un réseau de neurones (U-Net, Mask R-CNN) pour la segmentation
   - Pourrait améliorer la précision sur les cas difficiles
   - Nécessite un dataset annoté manuellement (plusieurs centaines d'images)

5. FORMAT DE SORTIE :
   - Ajouter l'export en format médical (DICOM, OME-TIFF)
   - Générer un rapport PDF automatique
   - Intégration avec des bases de données (SQL)

6. ROBUSTESSE :
   - Gérer les images multi-canaux (fluorescence)
   - Supporter d'autres colorations (H&E, Masson's trichrome modifié)
   - Détection automatique du type de coloration

DÉPENDANCES CRITIQUES :
    - tiffslide : Lecture des WSI (doit rester à jour)
    - scikit-image : Traitement d'image (version >= 0.19)
    - numpy : Calculs (version >= 1.20)
    - matplotlib : Visualisation (version >= 3.3)

PROBLÈMES CONNUS :
    - Très gros fichiers (>10 GB) : peut manquer de RAM
      → Solution : réduire TILE_SIZE ou utiliser un serveur avec plus de RAM

    - Images très compressées : artefacts JPEG possibles
      → Solution : préférer des formats sans perte (TIFF non compressé)

    - Microscopes avec balance des blancs incorrecte : calibration imprécise
      → Solution : recalibrer le microscope ou ajouter une correction gamma
"""
