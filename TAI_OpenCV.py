"""
TAI_V51_OpenCV.py - Analyse Automatisée d'Images Histologiques (version OpenCV)
===============================================================================
Analyse de coupes pulmonaires colorées au Trichrome de Masson pour quantifier :
    - Collagène (bleu)
    - Tissu normal (rose/rouge)
    - Air alvéolaire utile (blanc)

Dépendances principales : OpenCV (cv2), numpy, pandas

Cette version est une refonte du script TAI.py original pour utiliser 
exclusivement OpenCV pour le traitement d'image, en remplacement de
skimage, scipy et matplotlib.

Moteur : OpenCV + TiffSlide
Fonctionnalités : 
  - Calibration V36 (Force Cyan)
  - Smart Edges (Détection précise des bords)
  - Génération complète des images (Overlay, Masques, Debug)


"""

import os
import sys
import time
import csv
import glob
import datetime
import gc  # Ramasse-miettes pour la mémoire
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Barre de progression pour terminal

# Force matplotlib à ne pas ouvrir de fenêtres (évite les bugs graphiques)
plt.switch_backend('Agg')

# --- VÉRIFICATION DES DÉPENDANCES ---
try:
    import tiffslide
    import cv2
except ImportError as e:
    print(f"❌ ERREUR CRITIQUE : Il manque une librairie : {e}")
    print("👉 Exécutez : pip install opencv-python tiffslide matplotlib tqdm numpy")
    sys.exit(1)

# --- CONSTANTES & PARAMÈTRES ---
CODE_VERSION = "TAI v51 (Full Visual OpenCV)"
DATE_RUN = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

TILE_SIZE = 2048        # Taille des tuiles analysées
VIS_SC = 64             # Facteur de réduction pour la carte visuelle globale (économise la RAM)

# Seuils HSV (Normalisés 0.0 - 1.0)
HSV_SAT_MIN = 0.05      # Ignorer le gris/blanc sale
HSV_VAL_MIN = 0.10      # Ignorer le noir profond
HSV_VAL_MAX = 0.95      # Ignorer le blanc pur (lumière)

# Identifiants numériques
ID_AIR, ID_COLL, ID_TISS = 0, 1, 2

# Couleurs pour la visualisation (Format BGR pour OpenCV)
# Bleu pour collagène, Rouge pour tissu
COLOR_COLL = (255, 0, 0)   
COLOR_TISS = (0, 0, 255)   
COLOR_BG   = (255, 255, 255)

# =========================================================
# UTILITAIRES (Conversions & Normalisation)
# =========================================================

def to_normalized_hsv(img_rgb):
    """
    Convertit une image RGB en HSV via OpenCV.
    Normalise les valeurs entre 0.0 et 1.0 pour correspondre à la logique mathématique.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,0] /= 179.0 # Teinte (Hue)
    hsv[:,:,1] /= 255.0 # Saturation
    hsv[:,:,2] /= 255.0 # Valeur (Value)
    return hsv

def force_normalize_8bit(tile):
    """
    S'assure que l'image est bien en 8-bits (0-255) pour OpenCV.
    Gère les images 16-bits ou flottantes.
    """
    if tile is None or tile.size == 0: return None
    if tile.dtype == np.uint8: return tile
    img = tile.astype(float)
    if img.max() == img.min(): return np.zeros(tile.shape, dtype=np.uint8)
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)

# =========================================================
# 1. CALIBRATION (V36 - Cyan Force)
# =========================================================

def auto_calibrate_hsv_v36(slide, out_dir):
    """
    Détermine automatiquement la plage de couleur du bleu (collagène).
    Génère un graphique de contrôle.
    """
    print("   ↳ 🧠 Calibration (V36 - Force Cyan)...")
    
    # Récupération d'une miniature pour l'analyse globale
    try: thumb = np.array(slide.get_thumbnail((1024, 1024)).convert("RGB"))
    except: thumb = np.array(slide.get_thumbnail((512, 512)).convert("RGB"))

    # Conversion HSV
    hsv = to_normalized_hsv(thumb)
    hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # Filtrage des pixels pertinents (colorés)
    valid_mask = (sat > 0.15) & (val > 0.15) & (val < 0.95)
    valid_hues = hue[valid_mask]

    # Valeurs par défaut si image vide
    if len(valid_hues) == 0: return 0.48, 0.85

    # Histogramme des teintes
    hist, bin_edges = np.histogram(valid_hues, bins=120, range=(0, 1))
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Isolation de la zone bleue théorique
    blue_zone_mask = (centers > 0.35) & (centers < 0.85)
    hist_blue = hist.copy()
    hist_blue[~blue_zone_mask] = 0

    if np.sum(hist_blue) == 0: return 0.48, 0.85

    # Détection du pic principal
    peak_idx = np.argmax(hist_blue)
    threshold = np.max(hist_blue) * 0.15
    idx_max = peak_idx
    
    # On cherche la fin de la courbe bleue
    while idx_max < len(hist)-1 and hist[idx_max] > threshold and centers[idx_max] < 0.85:
        idx_max += 1
    detected_max = centers[idx_max]

    # --- Logique V36 ---
    # Min bloqué à 0.48 pour capturer le Cyan
    # Max capé à 0.85 pour ne pas prendre le violet (muscle)
    final_min = 0.48
    final_max = min(0.85, detected_max + 0.03)

    # Sauvegarde du graphique de calibration
    plt.figure(figsize=(10, 5))
    plt.plot(centers, hist, color='gray', alpha=0.5, label="Spectre complet")
    plt.axvspan(final_min, final_max, color='green', alpha=0.3, label="Zone Collagène")
    plt.title(f"Calibration V36 | {CODE_VERSION}")
    plt.xlabel("Teinte (Hue 0-1)")
    plt.savefig(os.path.join(out_dir, "0_CALIBRATION.png"))
    plt.close('all')

    return final_min, final_max

# =========================================================
# 2. GÉNÉRATION DES IMAGES (MOTEUR VISUEL)
# =========================================================

def create_visual_report(vis_map, slide, out_dir, counts):
    """
    Reconstruit les images finales à partir de la carte d'analyse (vis_map).
    Sauvegarde : Masque pur, Overlay (Transparence).
    """
    print("   ↳ 🎨 Génération des images finales...")
    
    H, W = vis_map.shape
    # Création d'une image vide couleur (BGR pour OpenCV)
    vis_color = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Colorisation selon les IDs
    vis_color[vis_map == ID_AIR] = [255, 255, 255] # Blanc
    vis_color[vis_map == ID_COLL] = COLOR_COLL     # Bleu
    vis_color[vis_map == ID_TISS] = COLOR_TISS     # Rouge

    # Sauvegarde du masque pur
    cv2.imwrite(os.path.join(out_dir, "VISUAL_MASK.png"), vis_color)

    # Création de l'Overlay (Superposition)
    try:
        # On récupère l'image originale à la même taille que le masque
        thumb = np.array(slide.get_thumbnail((W, H)).convert("RGB"))
        thumb = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
        thumb = cv2.resize(thumb, (W, H)) # Sécurité taille

        # Fusion : 70% Original + 30% Masque
        overlay = cv2.addWeighted(thumb, 0.7, vis_color, 0.3, 0)
        
        cv2.imwrite(os.path.join(out_dir, "VISUAL_OVERLAY.png"), overlay)
    except Exception as e:
        print(f"⚠️ Impossible de créer l'overlay : {e}")

# =========================================================
# 3. MOTEUR D'ANALYSE (OpenCV & Smart Edges)
# =========================================================

def create_radar_mask(slide):
    """
    Crée une carte grossière pour savoir où se trouve le tissu.
    Évite d'analyser le fond blanc inutilement.
    """
    print("   ↳ 📡 Radar : Scan de la structure...")
    try: thumb = np.array(slide.get_thumbnail((2048, 2048)).convert("RGB"))
    except: thumb = np.array(slide.get_thumbnail((1024, 1024)).convert("RGB"))
    
    thumb = force_normalize_8bit(thumb)
    gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    
    # Binarisation automatique (Otsu)
    thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Nettoyage : Suppression des poussières (< 500px)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    new_binary = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 500:
            new_binary[labels == i] = 255
            
    # Lissage des bords (Fermeture morphologique)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(new_binary, cv2.MORPH_CLOSE, kernel)
    
    # Remplissage des trous internes
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(closed)
    cv2.drawContours(filled_mask, contours, -1, 255, -1)
    
    return filled_mask > 0, thumb

def analyze_tile_opencv(tile, b_min, b_max):
    """
    Analyse pixel par pixel d'une tuile (Haute Définition).
    """
    hsv = to_normalized_hsv(tile)
    hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # 1. Détection Matière
    is_matter = (sat > HSV_SAT_MIN) & (val > HSV_VAL_MIN) & (val < HSV_VAL_MAX)
    
    # 2. Détection Collagène (selon calibration)
    is_coll = (hue >= b_min) & (hue <= b_max) & is_matter
    
    # 3. Détection Tissu (Le reste)
    is_tiss = is_matter & (~is_coll)

    mask = np.zeros(tile.shape[:2], dtype=np.uint8)
    mask[is_coll] = ID_COLL
    mask[is_tiss] = ID_TISS
    return mask

# =========================================================
# 4. WORKER (TRAITEMENT D'UNE IMAGE)
# =========================================================

def process_one_image(path, output_root):
    start_time = time.time()
    name = os.path.basename(path)
    print(f"\n🔹 TRAITEMENT : {name}")

    # Création du dossier de sortie spécifique
    out_dir = os.path.join(output_root, f"ANALYSE_{os.path.splitext(name)[0]}")
    os.makedirs(out_dir, exist_ok=True)

    slide = None
    try:
        # Ouverture de l'image (Robuste via TiffSlide)
        slide = tiffslide.TiffSlide(path)
        W, H = slide.dimensions

        # --- Initialisation de la carte visuelle ---
        # On divise par VIS_SC (ex: 64) pour que ça rentre dans la RAM
        vis_h, vis_w = H // VIS_SC, W // VIS_SC
        vis_map = np.zeros((vis_h, vis_w), dtype=np.uint8)

        # 1. Calibration
        bmin, bmax = auto_calibrate_hsv_v36(slide, out_dir)

        # 2. Radar
        mask_global, thumb = create_radar_mask(slide)
        hm, wm = mask_global.shape
        rx, ry = W/wm, H/hm

        # 3. Préparation des tâches
        tasks = []
        for y in range(0, H, TILE_SIZE):
            for x in range(0, W, TILE_SIZE):
                # Conversion coord HD -> coord Radar
                xs, xe = int(x/rx), int(min(x+TILE_SIZE, W)/rx)
                ys, ye = int(y/ry), int(min(y+TILE_SIZE, H)/ry)
                
                # On ajoute la tâche seulement si le radar voit du tissu
                if xs<wm and ys<hm and np.any(mask_global[ys:ye, xs:xe]):
                    tasks.append((x, y, xs, xe, ys, ye))

        # 4. Boucle de traitement
        counts = {ID_AIR: 0, ID_COLL: 0, ID_TISS: 0}
        
        print(f"   ↳ ⏳ Analyse détaillée : {len(tasks)} tuiles...")
        
        # tqdm affiche la barre de progression
        for i, (x, y, xs, xe, ys, ye) in enumerate(tqdm(tasks, unit="tuile")):
            wr, hr = min(TILE_SIZE, W - x), min(TILE_SIZE, H - y)
            
            # Lecture
            raw = slide.read_region((x, y), 0, (wr, hr))
            tile = np.array(raw.convert("RGB"))
            
            # Analyse HD
            mask_hd = analyze_tile_opencv(tile, bmin, bmax)
            
            # Application Smart Edges (Masque Radar interpolé)
            radar_patch = mask_global[ys:ye, xs:xe].astype(np.float32)
            if radar_patch.size == 0: continue
            
            radar_raw = cv2.resize(radar_patch, (wr, hr), interpolation=cv2.INTER_LINEAR)
            valid_loc = radar_raw > 0.15 # Seuil Smart Edges

            # Nettoyage du masque HD
            final_mask = mask_hd.copy()
            final_mask[~valid_loc] = ID_AIR

            # --- MISE À JOUR DE LA CARTE VISUELLE ---
            try:
                # Coordonnées dans la carte réduite
                vx, vy = x // VIS_SC, y // VIS_SC
                vw_t, vh_t = wr // VIS_SC, hr // VIS_SC
                
                if vw_t > 0 and vh_t > 0:
                    # Redimensionnement nearest neighbor (pas de flou de couleur)
                    mini_mask = cv2.resize(final_mask, (vw_t, vh_t), interpolation=cv2.INTER_NEAREST)
                    vis_map[vy:vy+vh_t, vx:vx+vw_t] = mini_mask
            except Exception:
                pass # Ignorer erreurs d'arrondi sur les bords extrêmes

            # --- COMPTAGE ---
            unique, u_counts = np.unique(final_mask[valid_loc], return_counts=True)
            for u, c in zip(unique, u_counts): counts[u] += c

            # --- DEBUG : Sauvegarder une image témoin au milieu ---
            if i == len(tasks) // 2: 
                 debug_img = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
                 # On peint directement sur l'image de debug
                 debug_img[final_mask == ID_COLL] = COLOR_COLL
                 debug_img[final_mask == ID_TISS] = COLOR_TISS
                 cv2.imwrite(os.path.join(out_dir, f"DEBUG_ZOOM_X{x}.jpg"), debug_img)

        # 5. Finalisation
        create_visual_report(vis_map, slide, out_dir, counts)

        # Calcul Score
        mat = counts[ID_COLL] + counts[ID_TISS]
        fib = (counts[ID_COLL] / mat * 100) if mat > 0 else 0
        
        # Sauvegarde TXT
        with open(os.path.join(out_dir, "resultats.txt"), "w") as f:
            f.write(f"Fichier: {name}\n")
            f.write(f"Fibrose: {fib:.2f}%\n")
            f.write(f"Collagene (px): {counts[ID_COLL]}\n")
            f.write(f"Tissu (px): {counts[ID_TISS]}\n")

        print(f"   ✅ TERMINE. Fibrose: {fib:.2f}%")
        
        return {
            "Fichier": name,
            "Date": DATE_RUN,
            "Fibrose (%)": round(fib, 2),
            "Collagene (px)": counts[ID_COLL],
            "Tissu (px)": counts[ID_TISS]
        }

    except Exception as e:
        print(f"❌ Erreur sur {name} : {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # Nettoyage mémoire obligatoire
        if slide: slide.close()
        del vis_map
        plt.close('all')
        gc.collect()

# =========================================================
# 5. INTERFACE UTILISATEUR (Menu Principal)
# =========================================================

def get_image_list():
    print("\n--- SÉLECTION DES IMAGES ---")
    print("1. Dossier complet (Tous les fichiers)")
    print("2. Fichier unique")
    choice = input("👉 Votre choix (1 ou 2) : ").strip()

    images = []
    if choice == "1":
        folder = input("Chemin du dossier : ").strip().replace('"', '')
        if os.path.isdir(folder):
            for ext in ("*.tif", "*.tiff", "*.ndpi", "*.svs", "*.mrxs", "*.scn", "*.jpg", "*.jpeg", "*.png", "*.bmp"):
                # Recherche insensible à la casse
                images.extend(glob.glob(os.path.join(folder, ext)))
                images.extend(glob.glob(os.path.join(folder, ext.upper())))
        else:
            print("❌ Dossier invalide.")
    elif choice == "2":
        f = input("Chemin du fichier : ").strip().replace('"', '')
        if os.path.isfile(f): images.append(f)
        else: print("❌ Fichier introuvable.")
    
    return sorted(list(set(images))) # Dédoublonnage et tri

def main():
    print("\n" + "="*50)
    print(f" {CODE_VERSION} ".center(50, "="))
    print("="*50)
    
    images = get_image_list()
    if not images:
        print("❌ Aucune image à traiter.")
        return

    out_root = input("Dossier de résultats (Entrée pour 'RESULTATS') : ").strip().replace('"', '')
    if not out_root: out_root = "RESULTATS"
    os.makedirs(out_root, exist_ok=True)

    results = []
    
    print(f"\n🚀 DÉMARRAGE DU BATCH ({len(images)} images)")
    
    for img_path in images:
        res = process_one_image(img_path, out_root)
        if res: results.append(res)
        # Petite pause pour laisser le disque souffler
        time.sleep(1)

    # Sauvegarde du CSV Global
    if results:
        csv_path = os.path.join(out_root, "RAPPORT_GLOBAL.csv")
        keys = results[0].keys()
        with open(csv_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, keys, delimiter=';')
            dict_writer.writeheader()
            dict_writer.writerows(results)
        print("\n" + "="*50)
        print(f"🎉 TOUT EST TERMINÉ. Rapport global : {csv_path}")
        print("="*50)

if __name__ == "__main__":
    main()
