"""
TAI_V51_ANNOTATED - CODE EXPLIQUÉ
=================================
Ce script est le moteur principal d'analyse.
Il a été commenté pour servir de documentation technique.
"""

import os
import sys
import time
import csv
import glob
import datetime
import gc  # Garbage Collector : Sert à vider la mémoire RAM manuellement
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Bibliothèque pour la barre de chargement

# Force matplotlib à ne pas ouvrir de fenêtres (évite les plantages sur serveur ou longues séries)
plt.switch_backend('Agg')

# --- CONSTANTES & INFO VERSION ---
CODE_VERSION = "TAI v51 (Smart Edges)"
DATE_RUN = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

try:
    # tiffslide est une alternative plus rapide et compatible que openslide pour les gros TIF
    import tiffslide
    from skimage import color, filters, morphology
    from skimage.transform import resize
    from scipy.ndimage import binary_fill_holes, center_of_mass
except ImportError as e:
    print(f"❌ Manque une librairie : {e}"); sys.exit(1)

# --- PARAMETRES DE SENSIBILITÉ ---
TILE_SIZE = 2048        # Taille des carrés analysés (compromis RAM/Vitesse)
HSV_SAT_MIN = 0.05      # Ignorer ce qui est trop gris (poussière, fond blanc sale)
HSV_VAL_MIN = 0.10      # Ignorer ce qui est trop noir
HSV_VAL_MAX = 0.95      # Ignorer ce qui est blanc pur (lumière microscope)

# Codes couleurs pour la visualisation (RGB normalisé 0-1)
ID_AIR, ID_COLL, ID_TISS = 0, 1, 2
C_AIR  = [0.9, 1.0, 0.9] # Vert pâle (Fond)
C_COLL = [0.0, 0.4, 1.0] # Bleu (Collagène)
C_TISS = [1.0, 0.2, 0.2] # Rouge (Muscle/Cellules)

# =========================================================
# 1. MOTEUR DE CALIBRATION (Cerveau de l'algo)
# =========================================================

def auto_calibrate_hsv_v36(slide, out_dir):
    """
    Cette fonction regarde l'image en basse résolution pour décider
    quelles nuances de bleu sont du collagène.
    """
    print("   ↳ 🧠 Calibration (Cyan Force)...")

    # On récupère une miniature (thumbnail) pour aller vite
    try: thumb = np.array(slide.get_thumbnail((1024, 1024)).convert("RGB"))
    except: thumb = np.array(slide.get_thumbnail((512, 512)).convert("RGB"))

    # Conversion en espace HSV (Teinte, Saturation, Valeur)
    # H (Hue) est le canal le plus important ici (la couleur pure)
    hsv = color.rgb2hsv(thumb)
    hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # On ne garde que les pixels colorés pour calculer l'histogramme
    valid_mask = (sat > 0.15) & (val > 0.15) & (val < 0.95)
    valid_hues = hue[valid_mask]

    # Sécurité si l'image est vide
    if len(valid_hues) == 0: return 0.48, 0.85

    # Calcul de la répartition des couleurs
    hist, bin_edges = np.histogram(valid_hues, bins=120, range=(0, 1))
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # On isole la zone "Bleu théorique" (0.35 à 0.85)
    blue_zone_mask = (centers > 0.35) & (centers < 0.85)
    hist_blue = hist.copy()
    hist_blue[~blue_zone_mask] = 0

    # Si pas de bleu, on renvoie les valeurs par défaut
    if np.sum(hist_blue) == 0: return 0.48, 0.85

    # Détection du sommet de la courbe bleue
    peak_idx = np.argmax(hist_blue)

    # Recherche de la fin du pic bleu (avant que ça ne devienne violet)
    threshold = np.max(hist_blue) * 0.15
    idx_max = peak_idx
    while idx_max < len(hist)-1 and hist[idx_max] > threshold and centers[idx_max] < 0.85:
        idx_max += 1
    detected_max = centers[idx_max]

    ### LOGIQUE "CYAN FORCE" (Spécifique V36/V51) ###
    # 1. Min = 0.48 : On force la prise en compte du Cyan (collagène clair/délavé)
    # 2. Max = 0.85 : On met un MUR indépassable pour ne jamais prendre le violet (muscle)
    final_min = 0.48
    final_max = min(0.85, detected_max + 0.03)

    # Génération du graphique de contrôle (0_CALIBRATION...)
    plt.figure(figsize=(10, 5))
    plt.plot(centers, hist, color='gray', alpha=0.5, label="Spectre complet")
    plt.axvspan(0.85, 1.0, color='red', alpha=0.1, label="Zone Tissu (>0.85)")
    plt.axvspan(final_min, final_max, color='green', alpha=0.3, label=f"Collagène ({final_min}-{final_max})")
    plt.title(f"Calibration V36 | {CODE_VERSION}")
    plt.savefig(os.path.join(out_dir, "0_CALIBRATION_V36.png"))
    plt.close('all')

    return final_min, final_max

# =========================================================
# 2. MOTEUR D'ANALYSE (Yeux de l'algo)
# =========================================================

def analyze_tile_adaptive(tile, b_min, b_max):
    """
    Analyse PIXEL PAR PIXEL d'une tuile haute définition.
    C'est ici que la précision se joue.
    """
    hsv = color.rgb2hsv(tile)
    hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # 1. Est-ce de la matière ? (Pas blanc, pas noir, pas gris)
    is_matter = (sat > HSV_SAT_MIN) & (val > HSV_VAL_MIN) & (val < HSV_VAL_MAX)

    # 2. Est-ce que la teinte est dans la fourchette "Collagène" calibrée plus haut ?
    is_coll = (hue >= b_min) & (hue <= b_max) & is_matter

    # 3. Tout ce qui est matière mais PAS collagène est du Tissu
    is_tiss = is_matter & (~is_coll)

    mask = np.zeros(tile.shape[:2], dtype=np.uint8)
    mask[is_coll] = ID_COLL; mask[is_tiss] = ID_TISS
    return mask

def force_normalize_8bit(tile):
    # Utilitaire pour s'assurer que l'image est lisible (convertit 16-bit en 8-bit si besoin)
    if tile is None or tile.size == 0: return None
    if tile.dtype == np.uint8: return tile
    img = tile.astype(float)
    if img.max() == img.min(): return np.zeros(tile.shape, dtype=np.uint8)
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)

def create_radar_mask(slide):
    """
    Crée une carte grossière du poumon pour savoir où regarder.
    Évite de traiter le fond blanc inutilement.
    """
    print("   ↳ 📡 Radar : Scan structure (Mode Smart V51)...")
    try: thumb = np.array(slide.get_thumbnail((2048, 2048)).convert("RGB"))
    except: thumb = np.array(slide.get_thumbnail((1024, 1024)).convert("RGB"))
    thumb = force_normalize_8bit(thumb)
    gray = color.rgb2gray(thumb)

    # Otsu : algorithme qui sépare automatiquement le fond clair du sujet sombre
    thresh = filters.threshold_otsu(gray)
    binary = gray < thresh # True là où il y a du tissu

    # Nettoyage (bouche les petits trous, supprime les poussières)
    binary = morphology.remove_small_objects(binary, min_size=500)
    binary = morphology.binary_closing(binary, morphology.disk(5))
    mask = binary_fill_holes(binary)

    return mask, thumb

# =========================================================
# 3. GESTION DES SORTIES (Rapports)
# =========================================================

def save_zoom_evidence(tile, mask, radar_hd, out_dir, x, y, stats):
    """ Sauvegarde une image témoin pour validation humaine """
    # Code de génération d'image (Omis pour brièveté, identique au code original)
    # ... (Création de la comparaison Originale vs Masque) ...
    pass

def generate_report(vis_map, counts, filename, out_dir, proc_time, bmin, bmax):
    """ Génère le rapport final PNG et calcule les scores """
    # Calcul des pourcentages
    total = sum(counts.values())
    if total == 0: total = 1

    # SCORE FIBROSE = Collagène / (Collagène + Tissu)
    # On exclut l'Air du calcul médical
    mat = counts[ID_COLL] + counts[ID_TISS]
    fib = (counts[ID_COLL] / mat * 100) if mat > 0 else 0

    # ... (Génération des graphiques Matplotlib) ...

    return {
        "Fichier": filename,
        "Date": DATE_RUN,
        "Version": CODE_VERSION,
        "Score_Fibrose": round(fib, 2),
        # ... autres stats ...
    }

# =========================================================
# 4. WORKER (L'ouvrier qui traite une image)
# =========================================================

def process_one_image(path, output_root):
    start = time.time()
    name = os.path.basename(path)
    print(f"\n🔹 FICHIER : {name}")

    # Création dossier spécifique
    out_dir = os.path.join(output_root, f"ANALYSE_{os.path.splitext(name)[0]}")
    os.makedirs(out_dir, exist_ok=True)

    slide = None
    try:
        slide = tiffslide.TiffSlide(path)

        # 1. Calibration
        bmin, bmax = auto_calibrate_hsv_v36(slide, out_dir)

        # 2. Radar
        W, H = slide.dimensions
        mask_global, thumb = create_radar_mask(slide)
        hm, wm = mask_global.shape
        # Ratios pour convertir coordonnées Radar -> Coordonnées HD
        rx, ry = W/wm, H/hm

        # 3. Création de la liste des tâches (Tuiles à scanner)
        tasks = []
        for y in range(0, H, TILE_SIZE):
            for x in range(0, W, TILE_SIZE):
                # On vérifie sur le radar si cette zone contient du tissu
                xs, xe = int(x/rx), int(min(x+TILE_SIZE, W)/rx)
                ys, ye = int(y/ry), int(min(y+TILE_SIZE, H)/ry)
                if xs<wm and ys<hm and np.any(mask_global[ys:ye, xs:xe]):
                    tasks.append((x, y, xs, xe, ys, ye))

        # ... (Initialisation compteurs) ...

        # Boucle principale de traitement
        for i, (x, y, xs, xe, ys, ye) in enumerate(tasks):
            # Lecture de la tuile HD
            raw = slide.read_region((x, y), 0, (wr, hr))
            tile = np.array(raw.convert("RGB"))

            # Analyse HD
            mask_hd = analyze_tile_adaptive(tile, bmin, bmax)

            ### CŒUR DE LA V51 : SMART EDGES ###
            # On redimensionne le masque radar basse définition vers la haute définition.
            # order=1 : Interpolation bilinéaire (crée un flou sur les bords au lieu d'escaliers)
            radar_raw = resize(mask_global[ys:ye, xs:xe].astype(float), (hr, wr), order=1, preserve_range=True)

            # Seuil 0.15 : C'est le compromis "Smart".
            # Le radar flou donne des valeurs entre 0 et 1 sur les bords.
            # > 0.15 permet de coller à la forme sans être trop strict (escalier) ni trop large (bruit).
            radar_loc = radar_raw > 0.15

            # On ne compte que les pixels valides (Détectés par analyse HD ET dans la zone Radar)
            valid = mask_hd[radar_loc]

            # ... (Mise à jour des compteurs et de la carte visuelle) ...

        # Fin de l'image
        data = generate_report(...)
        return data

    except Exception as e:
        print(f"❌ Erreur : {e}")
        return None

    finally:
        # Nettoyage IMPORTANT pour éviter le crash WinError 10055
        if slide: slide.close()
        plt.close('all')
        gc.collect() # Vide la RAM

# ... (Le reste est l'interface utilisateur Main UI déjà connue) ...
