"""
TAI.py - Analyse histologique de poumon (Trichrome de Masson)
==============================================================
Pipeline optimisé pour une utilisation rapide :
1. Détection du contour unique de l'échantillon
2. Quantification rapide (collagène, tissu, air utile)
Dépendances:
    pip install tifffile scikit-image imageio scipy matplotlib pandas numpy
Auteur: Projet TAI - Analyse SDRA
Date: 2025
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

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

def print_header(title):
    """Affiche un en-tête stylisé dans la console."""
    print("\n" + "="*70)
    print(f" {title.center(66)} ")
    print("="*70)

def print_info(message):
    """Affiche un message d'information dans la console."""
    print(f"ℹ️  {message}")

def print_success(message):
    """Affiche un message de succès dans la console."""
    print(f"✅  {message}")

def print_warning(message):
    """Affiche un message d'avertissement dans la console."""
    print(f"⚠  {message}")

def print_error(message):
    """Affiche un message d'erreur dans la console."""
    print(f"❌  {message}")

# ================================================================
# === 1. LECTURE DU FICHIER TIFF ================================
# ================================================================

def read_tiff(path):
    """Lit un fichier TIFF et renvoie un array RGB uint8."""
    print_info(f"Lecture du fichier TIFF : {os.path.basename(path)}")
    if tiff is not None:
        try:
            arr = tiff.imread(path)
        except Exception as e:
            print_error(f"Erreur lors de la lecture avec tifffile: {e}")
            arr = None
    else:
        arr = None
    if arr is None and imageio is not None:
        try:
            arr = imageio.imread(path)
        except Exception as e:
            print_error(f"Erreur lors de la lecture avec imageio: {e}")
            arr = None
    if arr is None:
        raise RuntimeError("Impossible de lire le fichier TIFF. Assurez-vous que tifffile ou imageio est installé.")

    # Conversion en RGB si nécessaire
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] != 3:
        arr = arr[..., :3]
    # Conversion en uint8
    if arr.dtype != np.uint8:
        if exposure is not None:
            arr = exposure.rescale_intensity(arr, out_range='uint8').astype(np.uint8)
        else:
            arr = ((arr - arr.min()) / (arr.ptp() + 1e-9) * 255).astype(np.uint8)
    print_success(f"Image chargée avec succès ({arr.shape[1]}x{arr.shape[0]} pixels, {arr.size//3/1e6:.2f} Mpx)")
    return arr

# ================================================================
# === 2. DÉTECTION DU CONTOUR EXTERNE ===========================
# ================================================================

def detect_sample_contour(img, subsample_factor=10, blur_sigma=2):
    """
    Détection rapide du contour externe principal.
    """
    print_info("Détection du contour principal en cours...")
    start = time.time()
    H, W = img.shape[:2]

    # Sous-échantillonnage pour accélérer
    img_small = img[::subsample_factor, ::subsample_factor]

    # Conversion en niveaux de gris
    gray = np.mean(img_small.astype(np.float32), axis=2) / 255.0

    # Floutage
    if gaussian_filter is not None:
        gray_blur = gaussian_filter(gray, sigma=blur_sigma)
    else:
        gray_blur = gray

    # Seuillage automatique (Otsu)
    thresh = filters.threshold_otsu(gray_blur)
    mask_small = gray_blur < thresh  # Tissu = sombre

    # Extraction du plus grand objet
    labeled = measure.label(mask_small)
    props = measure.regionprops(labeled)
    if not props:
        raise ValueError("Aucune région détectée dans l'image.")
    largest = max(props, key=lambda r: r.area)
    samplemask_small = labeled == largest.label

    # Nettoyage morphologique
    samplemask_small = morphology.binary_closing(samplemask_small, morphology.disk(5))
    samplemask_small = morphology.binary_dilation(samplemask_small, morphology.disk(2))

    # Remplir les trous internes
    if binary_fill_holes is not None:
        samplemask_small = binary_fill_holes(samplemask_small)

    # Remise à l'échelle du masque
    samplemask = resize(
        samplemask_small.astype(np.float32),
        (H, W),
        order=0,
        preserve_range=True,
        anti_aliasing=False
    ) > 0.5

    sample_pixels = samplemask.sum()
    excluded_pixels = (H * W) - sample_pixels
    print_success(f"Contour détecté en {time.time()-start:.1f}s")
    print(f"   - Zone à analyser: {sample_pixels:,} pixels ({sample_pixels/(H*W)*100:.1f}%)")
    print(f"   - Zone exclue: {excluded_pixels:,} pixels ({excluded_pixels/(H*W)*100:.1f}%)")

    return samplemask

# ================================================================
# === 3. QUANTIFICATION RAPIDE =================================
# ================================================================

def quantify_structures(img, samplemask, downscale_factor=5):
    """
    Quantifie collagène / tissu / air dans la zone utile (DEDANS le contour).
    Version avec sous-échantillonnage plus agressif (facteur 5).
    """
    print_info("Quantification des structures en cours...")
    start = time.time()

    # Sous-échantillonnage direct avec facteur 5
    small_img = img[::downscale_factor, ::downscale_factor].copy()
    small_mask = samplemask[::downscale_factor, ::downscale_factor].copy()

    print(f"   Image réduite: {small_img.shape[1]}x{small_img.shape[0]} pixels")

    # Conversion en HSV pour la détection de couleurs
    region = small_img.astype(np.float32) / 255.0
    hsv = color.rgb2hsv(region)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # === CRITÈRES DE DÉTECTION PAR COULEUR ===
    # Air: très clair (haute valeur) et peu saturé
    mask_air = (V > 0.85) & (S < 0.25) & small_mask
    # Collagène: bleu (H entre 0.55 et 0.75 en HSV = 198-270°)
    mask_collagen = ((H > 0.55) & (H < 0.75)) & (S > 0.25) & (V > 0.3) & small_mask
    # Tissu: rouge/rose (H proche de 0 ou 1 = 0-18° ou 342-360°)
    mask_tissue = ((H < 0.05) | (H > 0.9)) & (S > 0.25) & (V > 0.3) & small_mask

    total_pixels = small_mask.sum()
    if total_pixels == 0:
        print_warning("Aucun pixel échantillon détecté!")
        return {
            "Collagène (%)": 0,
            "Tissu (%)": 0,
            "Air utile (%)": 0
        }

    # Calcul des pourcentages
    results = {
        "Collagène (%)": (mask_collagen.sum() / total_pixels) * 100,
        "Tissu (%)": (mask_tissue.sum() / total_pixels) * 100,
        "Air utile (%)": (mask_air.sum() / total_pixels) * 100
    }

    print_success(f"Quantification terminée en {time.time()-start:.1f}s")
    print(f"   - Collagène (bleu):  {results['Collagène (%)']:.2f}%")
    print(f"   - Tissu (rose):      {results['Tissu (%)']:.2f}%")
    print(f"   - Air utile (blanc): {results['Air utile (%)']:.2f}%")

    return results

# ================================================================
# === 4. PIPELINE PRINCIPAL ====================================
# ================================================================

def main(image_path, output_dir=None, subsample_factor=10, blur_sigma=2, downscale_factor=5):
    """
    Pipeline complet d'analyse.
    """
    t0 = time.time()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Dossier de sortie
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    print_header("DÉBUT DU TRAITEMENT")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Dossier de sortie: {output_dir}")
    print(f"Heure de début: {start_time}")
    print("-"*70)

    # === ÉTAPE 1: LECTURE ===
    img = read_tiff(image_path)

    # === ÉTAPE 2: DÉTECTION DU CONTOUR ===
    samplemask = detect_sample_contour(img, subsample_factor, blur_sigma)

    # === ÉTAPE 3: QUANTIFICATION ===
    results = quantify_structures(img, samplemask, downscale_factor)

    # === ÉTAPE 4: EXPORT CSV ===
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration = time.time() - t0

    # Ajout des métadonnées aux résultats
    results_with_metadata = {
        "Nom du fichier": os.path.basename(image_path),
        "Chemin du fichier": image_path,
        "Date de traitement": start_time,
        "Heure de fin": end_time,
        "Durée du traitement (s)": duration,
        "Collagène (%)": results["Collagène (%)"],
        "Tissu (%)": results["Tissu (%)"],
        "Air utile (%)": results["Air utile (%)"],
        "Facteur de sous-échantillonnage": downscale_factor,
        "Taille originale (pixels)": f"{img.shape[1]}x{img.shape[0]}",
        "Taille réduite (pixels)": f"{img.shape[1]//downscale_factor}x{img.shape[0]//downscale_factor}"
    }

    csv_path = os.path.join(output_dir, f"{basename}_quantification.csv")
    pd.DataFrame([results_with_metadata]).to_csv(csv_path, index=False)

    print_success(f"Résultats exportés : {csv_path}")

    # === RÉSUMÉ FINAL ===
    print_header("TRAITEMENT TERMINÉ")
    print(f"Durée totale: {duration:.1f}s")
    print(f"Heure de fin: {end_time}")
    print("\nRÉSULTATS:")
    print(f"  - Collagène (fibrose):  {results['Collagène (%)']:.2f}%")
    print(f"  - Tissu normal:         {results['Tissu (%)']:.2f}%")
    print(f"  - Air utile:            {results['Air utile (%)']:.2f}%")
    print("\nFICHIERS GÉNÉRÉS:")
    print(f"  1. Résultats CSV:      {os.path.basename(csv_path)}")
    print("\n" + "="*70)

    return results_with_metadata

# ================================================================
# === 5. INTERFACE UTILISATEUR =================================
# ================================================================

def user_interface():
    """Interface utilisateur pour faciliter l'utilisation du script."""
    print_header("ANALYSE HISTOLOGIQUE DE POUmon (TRICHROME DE MASSON)")
    print("Ce programme permet d'analyser des images de poumon colorées au Trichrome de Masson.")
    print("Il détecte et quantifie le collagène (fibrose), le tissu normal et l'air utile.")

    # Demander le chemin de l'image
    while True:
        image_path = input("\nVeuillez entrer le chemin complet de votre image TIFF : ").strip()
        if os.path.isfile(image_path) and image_path.lower().endswith(('.tif', '.tiff')):
            break
        print_error("Le fichier spécifié n'existe pas ou n'est pas un fichier TIFF. Veuillez réessayer.")

    # Demander le dossier de sortie
    default_output_dir = os.path.dirname(image_path)
    output_dir = input(f"\nVeuillez entrer le dossier de sortie (appuyez sur Entrée pour utiliser {default_output_dir}) : ").strip()
    if not output_dir:
        output_dir = default_output_dir

    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    print_header("DÉMARRAGE DE L'ANALYSE")
    print_info("Veuillez patienter pendant le traitement de l'image...")

    # Lancer l'analyse
    try:
        results = main(image_path, output_dir)
        print_success("Analyse terminée avec succès!")
    except Exception as e:
        print_error(f"Une erreur est survenue pendant l'analyse: {e}")

    input("\nAppuyez sur Entrée pour quitter...")

# ================================================================
# === 6. EXÉCUTION ==============================================
# ================================================================

if __name__ == "__main__":
    user_interface()




