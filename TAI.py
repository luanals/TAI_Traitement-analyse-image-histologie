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

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

try:
    import tifffile as tiff
except ImportError:
    tiff = None

try:
    from skimage import filters, exposure, morphology, measure, color
    from skimage.transform import resize
except ImportError:
    filters = exposure = morphology = measure = color = resize = None

try:
    import imageio
except ImportError:
    imageio = None

try:
    from scipy.ndimage import gaussian_filter, binary_fill_holes
except ImportError:
    gaussian_filter = binary_fill_holes = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ================================================================
# AFFICHAGE
# ================================================================
def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title.center(66)} ")
    print("=" * 70)

def print_info(msg): print(f"ℹ️  {msg}")
def print_success(msg): print(f"✅  {msg}")
def print_warning(msg): print(f"⚠️  {msg}")
def print_error(msg): print(f"❌  {msg}")


# ================================================================
# LECTURE D’IMAGE AVEC SOUS-ÉCHANTILLONNAGE
# ================================================================
def read_tiff(path, downscale_preview=2):
    """
    Lit un TIFF avec sous-échantillonnage léger pour économiser la mémoire.
    downscale_preview = 2 → prend 1 pixel sur 2 dans chaque direction.
    """
    print_info(f"Lecture du fichier: {os.path.basename(path)}")

    arr = None
    if tiff is not None:
        try:
            arr = tiff.imread(path)
        except Exception as e:
            print_warning(f"tifffile a échoué: {e}")

    if arr is None and imageio is not None:
        arr = imageio.imread(path)

    if arr is None:
        raise RuntimeError("Impossible de lire le TIFF.")

    if downscale_preview > 1:
        H, W = arr.shape[:2]
        arr = arr[::downscale_preview, ::downscale_preview]
        print(f"   Image sous-échantillonnée ({downscale_preview}x) → {arr.shape[1]}x{arr.shape[0]} px")

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    if arr.dtype != np.uint8:
        if exposure is not None:
            arr = exposure.rescale_intensity(arr, out_range='uint8').astype(np.uint8)
        else:
            arr = ((arr - arr.min()) / (arr.ptp() + 1e-9) * 255).astype(np.uint8)

    print_success(f"Image chargée: {arr.shape[1]}x{arr.shape[0]} px")
    return arr


# ================================================================
# DÉTECTION DU CONTOUR
# ================================================================
def detect_sample_contour(img, subsample_factor=10, blur_sigma=2):
    print_info("Détection du contour en cours...")
    start = time.time()
    H, W = img.shape[:2]

    img_small = img[::subsample_factor, ::subsample_factor]
    gray = np.mean(img_small.astype(np.float32), axis=2) / 255.0

    if gaussian_filter is not None:
        gray_blur = gaussian_filter(gray, sigma=blur_sigma)
    else:
        gray_blur = gray

    thresh = filters.threshold_otsu(gray_blur)
    mask_small = gray_blur < thresh

    labeled = measure.label(mask_small)
    props = measure.regionprops(labeled)
    if not props:
        raise ValueError("Aucune région détectée.")

    largest = max(props, key=lambda r: r.area)
    samplemask_small = labeled == largest.label
    samplemask_small = morphology.binary_closing(samplemask_small, morphology.disk(5))
    samplemask_small = morphology.binary_dilation(samplemask_small, morphology.disk(2))
    if binary_fill_holes is not None:
        samplemask_small = binary_fill_holes(samplemask_small)

    samplemask = resize(samplemask_small.astype(np.float32), (H, W), order=0, preserve_range=True) > 0.5
    print_success(f"Contour détecté en {time.time() - start:.1f}s")
    return samplemask


# ================================================================
# QUANTIFICATION
# ================================================================
def quantify_structures(img, samplemask, downscale_factor=10):
    print_info("Quantification en cours...")
    start = time.time()

    small_img = img[::downscale_factor, ::downscale_factor]
    small_mask = samplemask[::downscale_factor, ::downscale_factor]

    hsv = color.rgb2hsv(small_img.astype(np.float32) / 255.0)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    mask_air = (V > 0.85) & (S < 0.25) & small_mask
    mask_collagen = ((H > 0.55) & (H < 0.75)) & (S > 0.25) & (V > 0.3) & small_mask
    mask_tissue = ((H < 0.05) | (H > 0.9)) & (S > 0.25) & (V > 0.3) & small_mask

    total_pixels = small_mask.sum()
    if total_pixels == 0:
        return {"Collagène (%)": 0, "Tissu (%)": 0, "Air utile (%)": 0}

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
# VISUALISATION
# ================================================================
def visualize_contour_zone(img, samplemask, output_path, timestamp):
    if plt is None:
        return
    display_factor = 20 if min(img.shape[:2]) > 5000 else 5
    img_d = img[::display_factor, ::display_factor]
    mask_d = samplemask[::display_factor, ::display_factor]
    overlay = img_d.copy().astype(np.float32)
    overlay[~mask_d] = 0.3 * overlay[~mask_d] + 0.7 * np.array([255, 255, 0])

    plt.figure(figsize=(10, 10))
    plt.imshow(overlay.astype(np.uint8))
    plt.axis("off")
    plt.title(f"Contour détecté (jaune = exclu)\n{timestamp}", fontsize=10)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_segmentation(img, samplemask, downscale_factor, output_path, timestamp):
    if plt is None or color is None:
        return
    small_img = img[::downscale_factor, ::downscale_factor]
    small_mask = samplemask[::downscale_factor, ::downscale_factor]
    hsv = color.rgb2hsv(small_img.astype(np.float32) / 255.0)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask_air = (V > 0.85) & (S < 0.25) & small_mask
    mask_collagen = ((H > 0.55) & (H < 0.75)) & (S > 0.25) & (V > 0.3) & small_mask
    mask_tissue = ((H < 0.05) | (H > 0.9)) & (S > 0.25) & (V > 0.3) & small_mask

    seg = np.zeros_like(small_img)
    seg[..., 0][mask_tissue] = 255
    seg[..., 1][mask_collagen] = 255
    seg[..., 2][mask_air] = 255

    plt.figure(figsize=(10, 10))
    plt.imshow(seg)
    plt.axis("off")
    plt.title(f"Segmentation - Rouge=Tissu | Vert=Collagène | Bleu=Air utile\n{timestamp}", fontsize=10)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ================================================================
# TRAITEMENT D’UNE IMAGE
# ================================================================
def process_single_image(image_path, output_folder, downscale_preview, downscale_factor):
    basename = Path(image_path).stem
    print_header(f"TRAITEMENT: {basename}")

    start_total = time.time()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    img = read_tiff(image_path, downscale_preview=downscale_preview)
    samplemask = detect_sample_contour(img, subsample_factor=10)

    contour_path = os.path.join(output_folder, f"{basename}_contour_{timestamp_str}.png")
    visualize_contour_zone(img, samplemask, contour_path, timestamp_str)

    seg_path = os.path.join(output_folder, f"{basename}_segmentation_{timestamp_str}.png")
    visualize_segmentation(img, samplemask, downscale_factor, seg_path, timestamp_str)

    results = quantify_structures(img, samplemask, downscale_factor)
    total_time = round(time.time() - start_total, 2)

    return {
        "Nom du fichier": basename,
        "Date/Heure": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **results,
        "Temps de calcul (s)": total_time,
        "Erreur": ""
    }


# ================================================================
# MODE D’ANALYSE
# ================================================================
def choose_analysis_mode():
    print_header("CHOIX DU MODE D'ANALYSE")
    print("1️⃣  Équilibré (recommandé) → preview=2, downscale=10")
    print("2️⃣  Mode sûr (anti-crash)  → preview=3, downscale=10")
    print("3️⃣  Haute précision        → preview=2, downscale=7")
    print("4️⃣  Batch rapide           → preview=3, downscale=12")

    while True:
        choice = input("👉 Choisissez un mode (1-4) : ").strip()
        if choice in ['1', '2', '3', '4']:
            break
        print_error("Choix invalide, réessayez.")

    return {"1": (2, 10), "2": (3, 10), "3": (2, 7), "4": (3, 12)}[choice]


# ================================================================
# BATCH AVEC CHOIX (manuel / dossier complet)
# ================================================================
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
        folder = input("Chemin du dossier contenant les images : ").strip()
        if not os.path.isdir(folder):
            print_error("Dossier introuvable.")
            return []
        for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
            images.extend(Path(folder).glob(ext))
        images = [str(p) for p in images]
        print_success(f"{len(images)} images détectées dans le dossier.")

    return images


# ================================================================
# MAIN
# ================================================================
def main():
    print_header("ANALYSE AUTOMATISÉE D'IMAGES HISTOLOGIQUES")
    downscale_preview, downscale_factor = choose_analysis_mode()

    # Test rapide sur une image
    print_header("PHASE 1: TEST SUR UNE IMAGE")
    test_image = input("Chemin de l'image de test (ou 'q' pour quitter): ").strip()
    if test_image.lower() == "q":
        print_warning("Analyse annulée.")
        return
    out_dir = os.path.join(os.path.dirname(test_image), "TEST_RESULTS")
    os.makedirs(out_dir, exist_ok=True)
    try:
        row = process_single_image(test_image, out_dir, downscale_preview, downscale_factor)
        print_success(f"Résultats du test : Collagène={row['Collagène (%)']}%, Tissu={row['Tissu (%)']}%, Air={row['Air utile (%)']}%")
    except Exception as e:
        print_error(f"Erreur pendant le test: {e}")
        return

    # Validation
    if input("Valider et passer au batch ? (o/n): ").strip().lower() != "o":
        print_warning("Analyse interrompue.")
        return

    # Phase batch
    print_header("PHASE 2: TRAITEMENT BATCH")
    images = get_image_list()
    if not images:
        print_error("Aucune image fournie.")
        return
    out_root = input("Dossier de sortie: ").strip()
    os.makedirs(out_root, exist_ok=True)

    all_rows = []
    for img_path in images:
        try:
            out_dir = os.path.join(out_root, Path(img_path).stem)
            os.makedirs(out_dir, exist_ok=True)
            row = process_single_image(img_path, out_dir, downscale_preview, downscale_factor)
            all_rows.append(row)
        except Exception as e:
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

    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(out_root, "ANALYSE_COMPLETE.csv")
        df.to_csv(csv_path, index=False)
        print_success(f"CSV global enregistré: {csv_path}")

        # Résumé automatique
        df_ok = df[df["Erreur"] == ""]
        if not df_ok.empty:
            print_header("RÉSUMÉ GLOBAL")
            print(f"→ Moyenne Collagène: {df_ok['Collagène (%)'].astype(float).mean():.2f}%")
            print(f"→ Moyenne Tissu: {df_ok['Tissu (%)'].astype(float).mean():.2f}%")
            print(f"→ Moyenne Air utile: {df_ok['Air utile (%)'].astype(float).mean():.2f}%")
            print(f"→ Temps moyen: {df_ok['Temps de calcul (s)'].astype(float).mean():.1f}s")

    print_header("FIN DU PROGRAMME")


if __name__ == "__main__":
    main()





