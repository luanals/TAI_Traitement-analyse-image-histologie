"""
TAI.py - Analyse Automatisée d'Images Histologiques (VERSION FINALE HSV)
=============================================================================
Analyse de coupes pulmonaires colorées au Trichrome de Masson pour quantifier :
  - Collagène (bleu)
  - Tissu normal (rose/rouge)
  - Air alvéolaire utile (blanc)

VERSION FINALE : Classification HSV optimisée sans OpenCV
"""

# ---------------------------
# IMPORTS ET GESTION D'ABSENCES
# ---------------------------
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from skimage.measure import label, regionprops

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
    from matplotlib.colors import rgb_to_hsv
except ImportError:
    plt = None
    rgb_to_hsv = None


# ================================================================
# ⚙️ PARAMÈTRES HSV AJUSTABLES (MODIFIEZ ICI)
# ================================================================
# Format HSV : H en [0, 1], S en [0, 1], V en [0, 1]
# H = 0.0 correspond à 0°, H = 1.0 correspond à 360°

# AIR (gris clair / blanc)
AIR_S_MAX = 0.25         # Saturation faible (quasi désaturé)
AIR_V_MIN = 0.75         # Luminosité haute

# COLLAGÈNE (bleu)
# Bleu en HSV : environ 200-250° soit 0.55-0.70 en [0,1]
COLLAGEN_H_MIN = 0.55    # ~198°
COLLAGEN_H_MAX = 0.70    # ~252°
COLLAGEN_S_MIN = 0.25    # Saturation élevée (bleu marqué)
COLLAGEN_V_MIN = 0.20    # Éviter bleu sombre/bruit

# TISSU/AUTRE (rose/rouge/violet/magenta)
# Rouge-violet : 270-360° et 0-36° soit 0.75-1.0 et 0.0-0.1
TISSUE_H_MIN1 = 0.75     # Magenta/violet haut (270°)
TISSUE_H_MAX1 = 1.00     # Rouge (360°)
TISSUE_H_MIN2 = 0.00     # Rouge (0°)
TISSUE_H_MAX2 = 0.10     # Orange/rouge (36°)
TISSUE_S_MIN = 0.20      # Saturation assez élevée
TISSUE_V_MIN = 0.15      # Luminosité minimale

# ================================================================


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


def print_hsv_guide():
    """Affiche un guide des paramètres HSV"""
    print_header("📊 GUIDE DE CALIBRATION HSV")
    print("Pour ajuster les seuils, modifiez les variables en haut du script")
    print("Format : H en [0, 1], S en [0, 1], V en [0, 1]")
    print("")
    print("AIR (gris clair / blanc) :")
    print(f"  Saturation max : {AIR_S_MAX}")
    print(f"  Luminosité min : {AIR_V_MIN}")
    print("")
    print("COLLAGÈNE (bleu) :")
    print(f"  Teinte H : {COLLAGEN_H_MIN} - {COLLAGEN_H_MAX} ({int(COLLAGEN_H_MIN*360)}° - {int(COLLAGEN_H_MAX*360)}°)")
    print(f"  Saturation min : {COLLAGEN_S_MIN}")
    print(f"  Luminosité min : {COLLAGEN_V_MIN}")
    print("")
    print("TISSU/AUTRE (rose/rouge/violet) :")
    print(f"  Teinte H : {TISSUE_H_MIN1}-{TISSUE_H_MAX1} ({int(TISSUE_H_MIN1*360)}°-{int(TISSUE_H_MAX1*360)}°)")
    print(f"           ou {TISSUE_H_MIN2}-{TISSUE_H_MAX2} ({int(TISSUE_H_MIN2*360)}°-{int(TISSUE_H_MAX2*360)}°)")
    print(f"  Saturation min : {TISSUE_S_MIN}")
    print(f"  Luminosité min : {TISSUE_V_MIN}")
    print("")
    print("💡 Pour convertir depuis un color picker RGB :")
    print("   Utilisez un convertisseur RGB→HSV en ligne")
    print("   Puis divisez H par 360 pour obtenir [0,1]")
    print("=" * 70)


# ================================================================
# LECTURE D'IMAGE
# ================================================================
def read_tiff(path, downscale_preview=2):
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
# CLASSIFICATION HSV (VOTRE FONCTION INTÉGRÉE)
# ================================================================
def classify_hsv(small_img, small_mask):
    """
    Classification HSV sans opencv (utilise matplotlib.colors.rgb_to_hsv).
    Retourne : mask_air, mask_collagen, mask_tissue_final, mask_unclassified
    """
    if rgb_to_hsv is None:
        raise ImportError("matplotlib.colors.rgb_to_hsv requis pour la conversion HSV")

    # Normalisation
    img_norm = small_img.astype(np.float32) / 255.0

    # Conversion HSV uniquement avec matplotlib
    hsv = rgb_to_hsv(img_norm)
    H = hsv[..., 0]   # 0–1
    S = hsv[..., 1]
    V = hsv[..., 2]

    # ==========================================================
    # 1) AIR (gris clair)
    # - teinte indéfinie
    # - faible saturation
    # - haute luminosité
    # ==========================================================
    mask_air = (
        (S < AIR_S_MAX) &
        (V > AIR_V_MIN) &
        small_mask
    )

    # ==========================================================
    # 2) COLLAGÈNE (bleu)
    # Bleu = H entre 0.55 et 0.70 environ en HSV matplotlib (≈200° à 250°)
    # Saturation élevée car bleu très marqué
    # ==========================================================
    mask_collagen = (
        (H >= COLLAGEN_H_MIN) & (H <= COLLAGEN_H_MAX) &   # fenêtre bleue
        (S > COLLAGEN_S_MIN) &                            # saturé
        (V > COLLAGEN_V_MIN) &                            # éviter bleu sombre ou bruit
        small_mask
    )

    # ==========================================================
    # 3) TISSU (rose / rouge / violet)
    # - teintes entre rouge→magenta→violet
    #   ~0.90–1.00 (rouge)  ou 0.00–0.10
    #   ~0.75–0.90 (magenta)
    # ==========================================================
    mask_tissue = (
        (
            ((H >= TISSUE_H_MIN1) & (H <= TISSUE_H_MAX1)) |   # violet/magenta/rouge haut
            ((H >= TISSUE_H_MIN2) & (H <= TISSUE_H_MAX2))     # rouge bas
        ) &
        (S > TISSUE_S_MIN) &                                  # doit être assez saturé (évite fond)
        (V > TISSUE_V_MIN) &
        small_mask
    )

    # ==========================================================
    # 4) NON CLASSÉS
    # ==========================================================
    mask_unclassified = small_mask & ~(mask_air | mask_collagen | mask_tissue)

    # On ajoute les non-classés dans "tissu" (reste biologique)
    mask_tissue_final = mask_tissue | mask_unclassified

    return mask_air, mask_collagen, mask_tissue_final, mask_unclassified


# ================================================================
# QUANTIFICATION (UTILISE classify_hsv)
# ================================================================
def quantify_structures(img, samplemask, downscale_factor=10):
    print_info("Quantification en cours...")
    start = time.time()

    # Sous-échantillonnage
    small_img = img[::downscale_factor, ::downscale_factor]
    small_mask = samplemask[::downscale_factor, ::downscale_factor]

    # Classification HSV
    mask_air, mask_collagen, mask_tissue_final, mask_unclassified = classify_hsv(small_img, small_mask)

    # Calcul des pourcentages
    total_pixels = small_mask.sum()
    if total_pixels == 0:
        raise ValueError("Aucun pixel à analyser dans le contour détecté.")

    collagen_pct = round(mask_collagen.sum() / total_pixels * 100, 2)
    tissue_pct = round(mask_tissue_final.sum() / total_pixels * 100, 2)
    air_pct = round(mask_air.sum() / total_pixels * 100, 2)
    unclassified_pct = round(mask_unclassified.sum() / total_pixels * 100, 2)

    # Vérification
    sum_pct = collagen_pct + tissue_pct + air_pct
    print_info(f"Somme des pourcentages = {sum_pct:.2f}% (devrait être 100%)")
    print_info(f"Pixels non classés fusionnés dans 'Autre' : {unclassified_pct}%")

    print_success(f"Quantification terminée en {time.time() - start:.1f}s")

    return {
        "Collagène (%)": collagen_pct,
        "Autre (%)": tissue_pct,
        "Air utile (%)": air_pct
    }, small_img, small_mask, mask_air, mask_collagen, mask_tissue_final, mask_unclassified


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


def visualize_segmentation(small_img, small_mask, output_path, timestamp):
    if plt is None:
        return

    # Recalcul des masques pour la visualisation
    mask_air, mask_collagen, mask_tissue_final, _ = classify_hsv(small_img, small_mask)

    seg = np.zeros_like(small_img)
    seg[..., 0][mask_tissue_final] = 255  # Rouge
    seg[..., 1][mask_collagen] = 255      # Vert
    seg[..., 2][mask_air] = 255           # Bleu

    plt.figure(figsize=(10, 10))
    plt.imshow(seg)
    plt.axis("off")
    plt.title(f"Segmentation - Rouge=Autre | Vert=Collagène | Bleu=Air utile\n{timestamp}", fontsize=10)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_unclassified(small_img, small_mask, mask_air, mask_collagen, mask_tissue, mask_unclassified, output_path, timestamp):
    """
    Affiche une carte des pixels non classés à l'intérieur du contour.
    Les pixels non classés apparaissent en ROUGE vif.
    """
    if plt is None:
        print_warning("Matplotlib requis pour visualisation des pixels non classés.")
        return

    vis = small_img.copy().astype(np.uint8)
    vis[mask_unclassified] = np.array([255, 0, 0], dtype=np.uint8)

    unclassified_pct = (mask_unclassified.sum() / small_mask.sum() * 100) if small_mask.sum() > 0 else 0

    plt.figure(figsize=(10, 10))
    plt.imshow(vis)
    plt.title(f"Pixels NON classés (rouge) : {unclassified_pct:.1f}%\n{timestamp}", fontsize=10)
    plt.axis("off")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_confidence_map(small_mask, mask_air, mask_collagen, mask_tissue, mask_unclassified, output_path, timestamp):
    """
    Génère une carte de confiance :
        2 = confiance forte (catégorisation nette)
        1 = moyenne (catégorisation mais couleur ambiguë)
        0 = faible (non classés)
    """
    if plt is None:
        print_warning("Matplotlib requis pour carte de confiance.")
        return

    confidence = np.zeros_like(small_mask, dtype=np.float32)
    confidence[mask_collagen | mask_air] = 2.0
    confidence[mask_tissue & ~mask_unclassified] = 1.0

    plt.figure(figsize=(10, 10))
    plt.imshow(confidence, cmap="inferno")
    plt.colorbar(label="Niveau de confiance (0–2)")
    plt.title(f"Carte de confiance de la segmentation\n{timestamp}", fontsize=10)
    plt.axis("off")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ================================================================
# TRAITEMENT D'UNE IMAGE
# ================================================================
def process_single_image(image_path, output_folder, downscale_preview, downscale_factor, include_diagnostics=False):
    """
    Traite une image et génère les visualisations.

    Args:
        include_diagnostics: Si True, génère aussi les cartes de confiance et pixels non classés
    """
    basename = Path(image_path).stem
    print_header(f"TRAITEMENT: {basename}")

    start_total = time.time()
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 1) Lecture
    img = read_tiff(image_path, downscale_preview=downscale_preview)

    # 2) Détection contour
    samplemask = detect_sample_contour(img, subsample_factor=10)

    # 3) Sauvegarde contour
    contour_path = os.path.join(output_folder, f"{basename}_contour_{timestamp_str}.png")
    visualize_contour_zone(img, samplemask, contour_path, timestamp_str)

    # 4) Quantification
    results, small_img, small_mask, mask_air, mask_collagen, mask_tissue, mask_unclassified = quantify_structures(
        img, samplemask, downscale_factor
    )

    # 5) Sauvegarde segmentation
    seg_path = os.path.join(output_folder, f"{basename}_segmentation_{timestamp_str}.png")
    visualize_segmentation(small_img, small_mask, seg_path, timestamp_str)

    # 6) OPTIONNEL : Visualisations de diagnostic
    if include_diagnostics:
        unclassified_path = os.path.join(output_folder, f"{basename}_unclassified_{timestamp_str}.png")
        visualize_unclassified(small_img, small_mask, mask_air, mask_collagen, mask_tissue, mask_unclassified,
                              unclassified_path, timestamp_str)

        confidence_path = os.path.join(output_folder, f"{basename}_confidence_{timestamp_str}.png")
        visualize_confidence_map(small_mask, mask_air, mask_collagen, mask_tissue, mask_unclassified,
                                confidence_path, timestamp_str)

    total_time = round(time.time() - start_total, 2)

    return {
        "Nom du fichier": basename,
        "Date/Heure": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **results,
        "Temps de calcul (s)": total_time,
        "Erreur": ""
    }


# ================================================================
# CHOIX DU MODE D'ANALYSE
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
# SÉLECTION DES IMAGES
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
            path = path.replace('"', '').replace("'", "")
            if os.path.isfile(path):
                images.append(path)
                print_success(f"Ajoutée: {os.path.basename(path)}")
            else:
                print_error("Fichier non trouvé.")
    else:
        folder = input("Chemin du dossier contenant les images : ").strip()
        folder = folder.replace('"', '').replace("'", "")
        if not os.path.isdir(folder):
            print_error("Dossier introuvable.")
            return []
        for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
            images.extend(Path(folder).glob(ext))
        images = [str(p) for p in images]
        print_success(f"{len(images)} images détectées dans le dossier.")

    return images


# ================================================================
# FONCTION MAIN
# ================================================================
def main():
    print_header("ANALYSE AUTOMATISÉE D'IMAGES HISTOLOGIQUES")
    print_hsv_guide()

    downscale_preview, downscale_factor = choose_analysis_mode()

    # CHOIX : TEST OU SKIP
    print_header("PHASE DE TEST")
    print("Voulez-vous faire un test sur une image avant le batch ?")
    print("1️⃣  Oui - Tester une image (recommandé)")
    print("2️⃣  Non - Passer directement au batch")

    while True:
        test_choice = input("👉 Votre choix (1 ou 2) : ").strip()
        if test_choice in ["1", "2"]:
            break
        print_error("Choix invalide.")

    # PHASE 1 : TEST (si choisi)
    if test_choice == "1":
        print_header("PHASE 1: TEST SUR UNE IMAGE")
        test_image = input("Chemin de l'image de test (ou 'q' pour quitter): ").strip()
        test_image = test_image.replace('"', '').replace("'", "")

        if test_image.lower() == "q":
            print_warning("Analyse annulée.")
            return

        out_dir = os.path.join(os.path.dirname(test_image), "TEST_RESULTS")
        os.makedirs(out_dir, exist_ok=True)
        try:
            row = process_single_image(test_image, out_dir, downscale_preview, downscale_factor, include_diagnostics=False)
            print_success(f"Résultats du test : Collagène={row['Collagène (%)']}%, Autre={row['Autre (%)']}%, Air={row['Air utile (%)']}%")
            print_info(f"Fichiers générés dans : {out_dir}")
        except Exception as e:
            print_error(f"Erreur pendant le test: {e}")
            return

        if input("Valider et passer au batch ? (o/n): ").strip().lower() != "o":
            print_warning("Analyse interrompue.")
            return

    # PHASE 2 : BATCH
    print_header("PHASE 2: TRAITEMENT BATCH")
    images = get_image_list()
    if not images:
        print_error("Aucune image fournie.")
        return

    out_root = input("Dossier de sortie: ").strip()
    out_root = out_root.replace('"', '').replace("'", "")
    os.makedirs(out_root, exist_ok=True)

    print("\nVoulez-vous générer les visualisations de diagnostic (pixels non classés + carte de confiance) ?")
    print("⚠️  Cela augmente le temps de traitement")
    diag_choice = input("👉 Générer les diagnostics ? (o/n) : ").strip().lower()
    include_diag = (diag_choice == "o")

    all_rows = []
    for img_path in images:
        try:
            out_dir = os.path.join(out_root, Path(img_path).stem)
            os.makedirs(out_dir, exist_ok=True)
            row = process_single_image(img_path, out_dir, downscale_preview, downscale_factor, include_diagnostics=include_diag)
            all_rows.append(row)
        except Exception as e:
            print_error(f"Erreur sur {img_path}: {e}")
            all_rows.append({
                "Nom du fichier": Path(img_path).stem,
                "Date/Heure": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Collagène (%)": "",
                "Autre (%)": "",
                "Air utile (%)": "",
                "Temps de calcul (s)": "",
                "Erreur": str(e)
            })

    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(out_root, "ANALYSE_COMPLETE.csv")
        df.to_csv(csv_path, index=False)
        print_success(f"CSV global enregistré: {csv_path}")

        df_ok = df[df["Erreur"] == ""]
        if not df_ok.empty:
            print_header("RÉSUMÉ GLOBAL")
            print(f"→ Moyenne Collagène: {df_ok['Collagène (%)'].astype(float).mean():.2f}%")
            print(f"→ Moyenne Autre: {df_ok['Autre (%)'].astype(float).mean():.2f}%")
            print(f"→ Moyenne Air utile: {df_ok['Air utile (%)'].astype(float).mean():.2f}%")
            print(f"→ Temps moyen: {df_ok['Temps de calcul (s)'].astype(float).mean():.1f}s")

    print_header("FIN DU PROGRAMME")

if __name__ == "__main__":
    main()








