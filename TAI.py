"""
TAI.py - Analyse Automatisée d'Images Histologiques (version finale complète)
=============================================================================
Analyse de coupes pulmonaires colorées au Trichrome de Masson pour quantifier :
  - Collagène (bleu)
  - Tissu normal (rose/rouge)
  - Air alvéolaire utile (blanc)

Optimisations :
  → Choix interactif du mode d’analyse (précision / vitesse / RAM)
  → Sous-échantillonnage à la lecture (downscale_preview)
  → Downscale factor configurable pour la quantification
  → Temps de calcul ajouté dans le CSV
  → Résultats arrondis à 2 chiffres significatifs
  → Gestion des erreurs : les images échouées apparaissent dans le CSV
  → Ajout de la date + légendes directement sur les figures
  → Choix entre sélection manuelle OU dossier complet pour le batch

Auteur : Projet TAI - Analyse SDRA (version finale optimisée)
Date : 2025
"""

# ---------------------------
# IMPORTS ET GESTION D'ABSENCES
# ---------------------------
# On importe des modules standards et on prépare des fallbacks si certaines bibliothèques
# ne sont pas disponibles sur la machine (afin d'avoir des messages d'erreur contrôlés).
import os                          # Pour manipuler chemins, dossiers, fichiers
import time                        # Pour mesurer durée des opérations
import numpy as np                 # Calcul numérique (tableaux, opérations vectorisées)
import pandas as pd                # Manipulation de tableaux / CSV (résultats)
from datetime import datetime      # Pour timestamp lisible
from pathlib import Path           # Pour gestion conviviale des chemins de fichiers

# tifffile (lecture efficace des TIFF) : on essaye d'importer, sinon on met tiff=None
try:
    import tifffile as tiff
except ImportError:
    tiff = None

# skimage (traitement d'image) : on essaye d'importer plusieurs modules. Si absent,
# on met les variables correspondantes à None et on gère cela plus loin.
try:
    from skimage import filters, exposure, morphology, measure, color
    from skimage.transform import resize
except ImportError:
    filters = exposure = morphology = measure = color = resize = None

# imageio : lecture alternative d'images
try:
    import imageio
except ImportError:
    imageio = None

# scipy.ndimage : filtrage Gaussien et remplissage de trous
try:
    from scipy.ndimage import gaussian_filter, binary_fill_holes
except ImportError:
    gaussian_filter = binary_fill_holes = None

# matplotlib.pyplot : pour sauver des figures (visualisation)
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ================================================================
# AFFICHAGE (fonctions utilitaires pour messages utilisateur)
# ================================================================
# Ces petites fonctions améliorent l'affichage en console pour l'utilisateur,
# ajoutent des séparateurs et des icônes pour repérer l'étape en cours.

def print_header(title):
    # Affiche un header encadré pour séparer visuellement les étapes
    print("\n" + "=" * 70)
    print(f" {title.center(66)} ")
    print("=" * 70)

def print_info(msg): print(f"ℹ️  {msg}")        # Message informatif
def print_success(msg): print(f"✅  {msg}")     # Message succès
def print_warning(msg): print(f"⚠️  {msg}")     # Mise en garde
def print_error(msg): print(f"❌  {msg}")       # Message d'erreur


# ================================================================
# LECTURE D’IMAGE AVEC SOUS-ÉCHANTILLONNAGE
# ================================================================
# Fonction responsable de la lecture des images (TIFF habituellement).
# Elle fait un sous-échantillonnage léger (preview) pour réduire usage mémoire
# et permet d'assurer que l'image est en uint8 (0-255) en sortie.

def read_tiff(path, downscale_preview=2):
    """
    Lit un TIFF avec sous-échantillonnage léger pour économiser la mémoire.
    - path : chemin vers le fichier image
    - downscale_preview : facteur entier (>1) pour prendre 1 pixel sur N (ex: 2 -> 1/2)
    Retour : un tableau numpy HxWx3 en uint8 (RGB)
    """
    # On affiche le nom du fichier en cours de lecture
    print_info(f"Lecture du fichier: {os.path.basename(path)}")

    arr = None  # variable qui contiendra finalement l'image lue

    # Si tifffile est disponible, on l'utilise (généralement plus robuste pour TIFF)
    if tiff is not None:
        try:
            arr = tiff.imread(path)
        except Exception as e:
            # Si la lecture tifffile échoue, on affiche un avertissement mais on continue
            print_warning(f"tifffile a échoué: {e}")

    # Si on n'a pas réussi à lire avec tifffile, on essaye imageio (lecture plus générique)
    if arr is None and imageio is not None:
        arr = imageio.imread(path)

    # Si aucune méthode n'a permis de lire l'image, on lève une erreur contrôlée
    if arr is None:
        raise RuntimeError("Impossible de lire le TIFF.")

    # Si on demande un sous-échantillonnage (downscale_preview > 1) :
    if downscale_preview > 1:
        H, W = arr.shape[:2]
        # On prend un pixel sur downscale_preview dans chaque direction
        arr = arr[::downscale_preview, ::downscale_preview]
        print(f"   Image sous-échantillonnée ({downscale_preview}x) → {arr.shape[1]}x{arr.shape[0]} px")

    # Si l'image est en niveaux de gris (ndim == 2), on la convertit en image RGB
    if arr.ndim == 2:
        # On empile 3 fois la même couche pour avoir shape (H,W,3)
        arr = np.stack([arr] * 3, axis=-1)

    # On s'assure que le format des pixels est uint8 (valeurs 0-255)
    # Certains fichiers TIFF sont en float ou en uint16, on normalise ici.
    if arr.dtype != np.uint8:
        if exposure is not None:
            # skimage.exposure.rescale_intensity permet une mise à l'échelle robuste
            arr = exposure.rescale_intensity(arr, out_range='uint8').astype(np.uint8)
        else:
            # fallback manuel si skimage manque : min-max scaling
            arr = ((arr - arr.min()) / (arr.ptp() + 1e-9) * 255).astype(np.uint8)

    # On retourne l'image prête à être traitée
    print_success(f"Image chargée: {arr.shape[1]}x{arr.shape[0]} px")
    return arr


# ================================================================
# DÉTECTION DU CONTOUR DE L'ÉCHANTILLON (MASQUE GLOBAL)
# ================================================================
# L'objectif : trouver la région principale de l'échantillon (la coupe) dans
# l'image afin d'exclure le fond et les zones hors échantillon.
# Étapes principales :
#  - sous-échantillonnage pour vitesse
#  - conversion en gris (moyenne des canaux)
#  - filtre gaussien (optionnel)
#  - seuillage automatique (Otsu)
#  - labellisation des régions et sélection de la plus grande
#  - opérations morphologiques pour lisser le masque
#  - rééchantillonnage du masque à la taille initiale

def detect_sample_contour(img, subsample_factor=10, blur_sigma=2):
    """
    Détecte le contour principal (masque binaire) d'un échantillon posé sur une lame.
    - img : image RGB numpy
    - subsample_factor : réduction utilisée pour la détection rapide (ex: 10)
    - blur_sigma : sigma du filtre gaussien appliqué au gris (si disponible)
    Retour : masque binaire de la taille originale (True = zone d'échantillon)
    """
    print_info("Détection du contour en cours...")
    start = time.time()  # pour mesurer la durée de l'opération
    H, W = img.shape[:2]

    # Sous-échantillonnage pour accélérer la détection du contour
    img_small = img[::subsample_factor, ::subsample_factor]

    # Conversion en "gris" simple : moyenne des trois canaux normalisée (0..1)
    gray = np.mean(img_small.astype(np.float32), axis=2) / 255.0

    # Application d'un flou gaussien si disponible (pour supprimer le bruit)
    if gaussian_filter is not None:
        gray_blur = gaussian_filter(gray, sigma=blur_sigma)
    else:
        gray_blur = gray

    # Calcul du seuil d'Otsu (méthode automatique pour séparer fore/ back)
    thresh = filters.threshold_otsu(gray_blur)
    # mask_small : True pour pixels sombres (ex. tissu) — dépend du contraste de la préparation
    mask_small = gray_blur < thresh

    # Labellisation des régions connectées (pour trouver la plus grande région = échantillon)
    labeled = measure.label(mask_small)
    props = measure.regionprops(labeled)
    if not props:
        # Si aucune région détectée, on signale une erreur maîtrisée
        raise ValueError("Aucune région détectée.")

    # On choisit la région ayant la plus grande aire (on suppose que c'est la coupe)
    largest = max(props, key=lambda r: r.area)
    samplemask_small = labeled == largest.label

    # On ferme et dilate le masque pour lisser les contours (morphologie binaire)
    samplemask_small = morphology.binary_closing(samplemask_small, morphology.disk(5))
    samplemask_small = morphology.binary_dilation(samplemask_small, morphology.disk(2))

    # On remplit les trous si la fonction scipy est disponible (pour avoir un masque plein)
    if binary_fill_holes is not None:
        samplemask_small = binary_fill_holes(samplemask_small)

    # On remet le masque à la taille d'origine (resize nearest / order=0 pour garder binaire)
    samplemask = resize(samplemask_small.astype(np.float32), (H, W), order=0, preserve_range=True) > 0.5

    print_success(f"Contour détecté en {time.time() - start:.1f}s")
    return samplemask


# ================================================================
# QUANTIFICATION DES STRUCTURES (SEUILLAGE DANS HSV)
# ================================================================
# Convertit l'image RGB réduite en HSV puis applique des conditions logiques
# sur les canaux H (Teinte), S (Saturation), V (Valeur) pour créer trois masques :
#  - mask_air      : espaces bleus/gris très clairs (V élevé, S bas)
#  - mask_collagen : plages de teinte correspondant au bleu (H entre ~0.55 et 0.75)
#  - mask_tissue   : plages rouge/rose (H proche de 0 ou proche de 1)
# On compte ensuite les pixels appartenant à chaque masque (seulement à l'intérieur du samplemask)
# et retourne des pourcentages.

def quantify_structures(img, samplemask, downscale_factor=10):
    """
    Quantifie Collagène, Tissu et Air utile.
    - img : image RGB full-size
    - samplemask : masque binaire full-size (True = zone à analyser)
    - downscale_factor : facteur pour réduire la résolution pendant la quantification
    Retour : dict avec clés "Collagène (%)", "Tissu (%)", "Air utile (%)"
    """
    print_info("Quantification en cours...")
    start = time.time()

    # On réduit l'image pour accélérer la quantification, en gardant le masque cohérent
    small_img = img[::downscale_factor, ::downscale_factor]
    small_mask = samplemask[::downscale_factor, ::downscale_factor]

    # Conversion RGB -> HSV (valeurs normalisées 0..1)
    hsv = color.rgb2hsv(small_img.astype(np.float32) / 255.0)
    # Séparation des canaux (H, S, V)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Définition des masques par conditions logiques :
    # - Air : pixels très clairs (V>0.85) et peu saturés (S<0.25)
    mask_air = (V > 0.85) & (S < 0.25) & small_mask

    # - Collagène : teinte bleue approximative (H entre 0.55 et 0.75), saturation suffisante, valeur min
    mask_collagen = ((H > 0.55) & (H < 0.75)) & (S > 0.25) & (V > 0.3) & small_mask

    # - Tissu : teinte rouge/rose (H proche de 0 ou proche de 1), saturation et valeur suffisantes
    mask_tissue = ((H < 0.05) | (H > 0.9)) & (S > 0.25) & (V > 0.3) & small_mask

    # Nombre total de pixels pertinents (dans le masque de l'échantillon)
    total_pixels = small_mask.sum()
    if total_pixels == 0:
        # Cas anormal mais possible : on retourne 0 pour éviter division par zéro
        return {"Collagène (%)": 0, "Tissu (%)": 0, "Air utile (%)": 0}

    # Calcul des pourcentages (arrondis à 2 décimales)
    collagen_pct = round(mask_collagen.sum() / total_pixels * 100, 2)
    tissue_pct = round(mask_tissue.sum() / total_pixels * 100, 2)
    air_pct = round(mask_air.sum() / total_pixels * 100, 2)

    results = {
        "Collagène (%)": collagen_pct,
        "Tissu (%)": tissue_pct,
        "Air utile (%)": air_pct
    }

    print_success(f"Quantification terminée en {time.time() - start:.1f}s")
    return results


# ================================================================
# VISUALISATION (sauvegarde d'images illustratives)
# ================================================================
# Deux fonctions : l'une pour afficher le contour détecté (overlay),
# l'autre pour afficher la segmentation colorée (rouge/vert/bleu).
# Ces visualisations sont sauvegardées en PNG dans le dossier de sortie.

def visualize_contour_zone(img, samplemask, output_path, timestamp):
    """
    Sauvegarde une image montrant le contour détecté en overlay.
    - img : image RGB
    - samplemask : masque binaire
    - output_path : chemin du fichier PNG à sauver
    - timestamp : chaîne date/heure ajoutée au titre
    """
    if plt is None:
        # Si matplotlib absent, on ne fait rien (fonction silencieuse)
        return

    # On adapte le facteur d'affichage en fonction de la taille pour ne pas créer d'images immenses
    display_factor = 20 if min(img.shape[:2]) > 5000 else 5
    img_d = img[::display_factor, ::display_factor]
    mask_d = samplemask[::display_factor, ::display_factor]

    # Création d'un overlay visuel : on assombrit/exclut les pixels hors masque en les teignant
    overlay = img_d.copy().astype(np.float32)
    overlay[~mask_d] = 0.3 * overlay[~mask_d] + 0.7 * np.array([255, 255, 0])
    # Le jaune (255,255,0) est utilisé pour marquer les zones exclues ; on mélange pour garder détail

    plt.figure(figsize=(10, 10))
    plt.imshow(overlay.astype(np.uint8))
    plt.axis("off")
    plt.title(f"Contour détecté (jaune = exclu)\n{timestamp}", fontsize=10)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_segmentation(img, samplemask, downscale_factor, output_path, timestamp):
    """
    Sauvegarde une image représentant la segmentation (R= tissu, V= collagène, B= air).
    - img : image RGB full-size
    - samplemask : masque binaire full-size
    - downscale_factor : facteur de réduction pour visualisation
    - output_path : chemin du fichier PNG
    - timestamp : chaîne date/heure ajoutée au titre
    """
    if plt is None or color is None:
        # Si manquant, on sort sans rien faire
        return

    # On réduit l'image pour la visualisation (évite images trop lourdes)
    small_img = img[::downscale_factor, ::downscale_factor]
    small_mask = samplemask[::downscale_factor, ::downscale_factor]
    hsv = color.rgb2hsv(small_img.astype(np.float32) / 255.0)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Recalcul des masques (mêmes critères que dans quantify_structures)
    mask_air = (V > 0.85) & (S < 0.25) & small_mask
    mask_collagen = ((H > 0.55) & (H < 0.75)) & (S > 0.25) & (V > 0.3) & small_mask
    mask_tissue = ((H < 0.05) | (H > 0.9)) & (S > 0.25) & (V > 0.3) & small_mask

    # Préparation d'une image RGB vide (noire) et remplissage des canaux pour visualiser
    seg = np.zeros_like(small_img)
    seg[..., 0][mask_tissue] = 255     # canal rouge = tissu
    seg[..., 1][mask_collagen] = 255  # canal vert = collagène
    seg[..., 2][mask_air] = 255       # canal bleu = air utile

    plt.figure(figsize=(10, 10))
    plt.imshow(seg)
    plt.axis("off")
    plt.title(f"Segmentation - Rouge=Tissu | Vert=Collagène | Bleu=Air utile\n{timestamp}", fontsize=10)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ================================================================
# TRAITEMENT D’UNE IMAGE (fonction orchestratrice)
# ================================================================
# Cette fonction exécute toutes les étapes pour une image donnée :
#  - lecture, détection contour, sauvegarde contour, sauvegarde segmentation,
#    quantification, et renvoi d'un dictionnaire de résultats.

def process_single_image(image_path, output_folder, downscale_preview, downscale_factor):
    """
    Processus complet pour une seule image :
    - image_path : chemin vers l'image
    - output_folder : dossier où sauvegarder les images de sortie
    - downscale_preview : facteur de sous-échantillonnage pour la lecture
    - downscale_factor : facteur utilisé pour la quantification et la segmentation visuelle
    Retour : dict prêt à être ajouté à un DataFrame / CSV
    """
    # basename = nom de fichier sans extension (utilisé pour construire noms de fichiers résultats)
    basename = Path(image_path).stem
    print_header(f"TRAITEMENT: {basename}")

    start_total = time.time()  # pour mesurer le temps total de traitement
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")  # timestamp pour fichiers

    # 1) Lecture de l'image (avec sous-échantillonnage preview)
    img = read_tiff(image_path, downscale_preview=downscale_preview)

    # 2) Détection du contour (masque)
    samplemask = detect_sample_contour(img, subsample_factor=10)

    # 3) Sauvegarde image overlay montrant le contour détecté
    contour_path = os.path.join(output_folder, f"{basename}_contour_{timestamp_str}.png")
    visualize_contour_zone(img, samplemask, contour_path, timestamp_str)

    # 4) Sauvegarde image de segmentation colorée
    seg_path = os.path.join(output_folder, f"{basename}_segmentation_{timestamp_str}.png")
    visualize_segmentation(img, samplemask, downscale_factor, seg_path, timestamp_str)

    # 5) Quantification effective (retourne un dict de pourcentages)
    results = quantify_structures(img, samplemask, downscale_factor)
    total_time = round(time.time() - start_total, 2)  # temps total arrondi

    # On retourne un dictionnaire avec les résultats et métadonnées
    return {
        "Nom du fichier": basename,
        "Date/Heure": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **results,
        "Temps de calcul (s)": total_time,
        "Erreur": ""   # champ vide = pas d'erreur
    }


# ================================================================
# CHOIX DU MODE D'ANALYSE (interaction utilisateur)
# ================================================================
# Fonction interactive qui propose des presets (équilibré, sûr, précision, batch rapide)
# Le preset renvoie deux valeurs : downscale_preview (lecture) et downscale_factor (quantification)

def choose_analysis_mode():
    print_header("CHOIX DU MODE D'ANALYSE")
    # Explication des presets disponibles
    print("1️⃣  Équilibré (recommandé) → preview=2, downscale=10")
    print("2️⃣  Mode sûr (anti-crash)  → preview=3, downscale=10")
    print("3️⃣  Haute précision        → preview=2, downscale=7")
    print("4️⃣  Batch rapide           → preview=3, downscale=12")

    # Boucle de validation de saisie utilisateur (on attend 1-4)
    while True:
        choice = input("👉 Choisissez un mode (1-4) : ").strip()
        if choice in ['1', '2', '3', '4']:
            break
        print_error("Choix invalide, réessayez.")

    # On retourne un tuple (downscale_preview, downscale_factor) selon le choix
    return {"1": (2, 10), "2": (3, 10), "3": (2, 7), "4": (3, 12)}[choice]


# ================================================================
# SÉLECTION DES IMAGES (manuel ou dossier complet)
# ================================================================
# Permet soit d'ajouter manuellement des fichiers un par un, soit de scanner
# un dossier et prendre toutes les images avec extension commune.

def get_image_list():
    print_header("SÉLECTION DES IMAGES")
    print("1️⃣  Sélection manuelle (une par une, avec 'done' à la fin)")
    print("2️⃣  Dossier complet (toutes les images du dossier)")
    while True:
        mode = input("👉 Choisissez un mode (1 ou 2) : ").strip()
        if mode in ["1", "2"]:
            break
        print_error("Choix invalide.")

    images = []
    if mode == "1":
        # Mode manuel : l'utilisateur tape des chemins, 'done' termine la liste
        while True:
            path = input("Chemin image (ou 'done'): ").strip()
            if path.lower() == "done":
                break
            if os.path.isfile(path):
                images.append(path)
                print_success(f"Ajoutée: {os.path.basename(path)}")
            else:
                print_error("Fichier non trouvé.")
    else:
        # Mode dossier complet : on prend toutes les images du dossier
        folder = input("Chemin du dossier contenant les images : ").strip()
        if not os.path.isdir(folder):
            print_error("Dossier introuvable.")
            return []
        # On cherche plusieurs extensions communes
        for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
            images.extend(Path(folder).glob(ext))
        # On convertit les Path en str
        images = [str(p) for p in images]
        print_success(f"{len(images)} images détectées dans le dossier.")

    return images


# ================================================================
# FONCTION MAIN (orchestration générale et interface utilisateur)
# ================================================================
#  - choix du preset
#  - test sur une image unique (phase 1)
#  - validation utilisateur
#  - batch (phase 2) avec sauvegarde CSV résumé

def main():
    print_header("ANALYSE AUTOMATISÉE D'IMAGES HISTOLOGIQUES")
    # On récupère le preset (downscale_preview, downscale_factor)
    downscale_preview, downscale_factor = choose_analysis_mode()

    # PHASE 1 : test sur une image (permet de vérifier paramètres avant le batch)
    print_header("PHASE 1: TEST SUR UNE IMAGE")
    test_image = input("Chemin de l'image de test (ou 'q' pour quitter): ").strip()
    if test_image.lower() == "q":
        print_warning("Analyse annulée.")
        return
    # Dossier de sortie par défaut pour le test : dans le même dossier que l'image
    out_dir = os.path.join(os.path.dirname(test_image), "TEST_RESULTS")
    os.makedirs(out_dir, exist_ok=True)
    try:
        # Exécution du traitement complet sur l'image test
        row = process_single_image(test_image, out_dir, downscale_preview, downscale_factor)
        # Affichage des résultats succincts
        print_success(f"Résultats du test : Collagène={row['Collagène (%)']}%, Tissu={row['Tissu (%)']}%, Air={row['Air utile (%)']}%")
    except Exception as e:
        # Si erreur lors du test, on la signale et on arrête le programme
        print_error(f"Erreur pendant le test: {e}")
        return

    # On demande une validation pour passer au traitement batch (évite erreurs massives)
    if input("Valider et passer au batch ? (o/n): ").strip().lower() != "o":
        print_warning("Analyse interrompue.")
        return

    # PHASE 2 : traitement batch
    print_header("PHASE 2: TRAITEMENT BATCH")
    images = get_image_list()
    if not images:
        print_error("Aucune image fournie.")
        return
    # On demande le dossier de sortie global
    out_root = input("Dossier de sortie: ").strip()
    os.makedirs(out_root, exist_ok=True)

    all_rows = []  # liste qui contiendra un dict (ligne) par image traitée
    for img_path in images:
        try:
            # Pour chaque image, on crée un sous-dossier nommé selon la base du fichier
            out_dir = os.path.join(out_root, Path(img_path).stem)
            os.makedirs(out_dir, exist_ok=True)
            # Traitement complet
            row = process_single_image(img_path, out_dir, downscale_preview, downscale_factor)
            all_rows.append(row)
        except Exception as e:
            # En cas d'erreur lors d'une image, on la journalise dans le CSV final
            print_error(f"Erreur sur {img_path}: {e}")
            all_rows.append({
                "Nom du fichier": Path(img_path).stem,
                "Date/Heure": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Collagène (%)": "",
                "Tissu (%)": "",
                "Air utile (%)": "",
                "Temps de calcul (s)": "",
                "Erreur": str(e)
            })

    # Si on a au moins une ligne, on crée un CSV récapitulatif
    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(out_root, "ANALYSE_COMPLETE.csv")
        df.to_csv(csv_path, index=False)
        print_success(f"CSV global enregistré: {csv_path}")

        # Résumé automatique : moyenne sur les images sans erreur
        df_ok = df[df["Erreur"] == ""]
        if not df_ok.empty:
            print_header("RÉSUMÉ GLOBAL")
            # On force la conversion en float pour calcul des moyennes
            print(f"→ Moyenne Collagène: {df_ok['Collagène (%)'].astype(float).mean():.2f}%")
            print(f"→ Moyenne Tissu: {df_ok['Tissu (%)'].astype(float).mean():.2f}%")
            print(f"→ Moyenne Air utile: {df_ok['Air utile (%)'].astype(float).mean():.2f}%")
            print(f"→ Temps moyen: {df_ok['Temps de calcul (s)'].astype(float).mean():.1f}s")

    print_header("FIN DU PROGRAMME")


# Si on exécute le script directement (python TAI.py), on lance main()
if __name__ == "__main__":
    main()






