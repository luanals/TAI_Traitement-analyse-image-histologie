"""
TAI_OpenCV.py - Analyse Automatisée d'Images Histologiques (version OpenCV)
=============================================================================
Analyse de coupes pulmonaires colorées au Trichrome de Masson pour quantifier :
    - Collagène (bleu)
    - Tissu normal (rose/rouge)
    - Air alvéolaire utile (blanc)

Dépendances principales : OpenCV (cv2), numpy, pandas

Cette version est une refonte du script TAI.py original pour utiliser 
exclusivement OpenCV pour le traitement d'image, en remplacement de
skimage, scipy, tifffile et matplotlib.

Auteur :  Projet TAI - Analyse SDRA (version finale optimisée OpenCV)
Date : 2025
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

# On vérifie les dépendances critiques (OpenCV, Numpy, Pandas)
try:
    import cv2
except ImportError:
    print("ERREUR: OpenCV (cv2) n'est pas installé. Veuillez l'installer avec 'pip install opencv-python'")
    exit()

try:
    import numpy as np
except ImportError:
    print("ERREUR: Numpy n'est pas installé. Veuillez l'installer avec 'pip install numpy'")
    exit()

try:
    import pandas as pd
except ImportError:
    print("ERREUR: Pandas n'est pas installé. Veuillez l'installer avec 'pip install pandas'")
    exit()


# ================================================================
# AFFICHAGE (fonctions utilitaires pour messages utilisateur)
# ================================================================
# (Identique à l'original)

def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title.center(66)} ")
    print("=" * 70)

def print_info(msg): print(f"ℹ️  {msg}")
def print_success(msg): print(f"✅  {msg}")
def print_warning(msg): print(f"⚠️  {msg}")
def print_error(msg): print(f"❌  {msg}")


# ================================================================
# LECTURE D’IMAGE AVEC SOUS-ÉCHANTILLONNAGE (via OpenCV)
# ================================================================
# Utilise cv2.imread pour lire tous formats (TIFF, PNG, JPG...)

def read_image_cv(path, downscale_preview=2):
    """
    Lit une image avec sous-échantillonnage léger pour économiser la mémoire.
    Utilise OpenCV.
    - path : chemin vers le fichier image
    - downscale_preview : facteur entier (>1) pour prendre 1 pixel sur N
    Retour : un tableau numpy HxWx3 en uint8 (RGB)
    """
    print_info(f"Lecture du fichier: {os.path.basename(path)}")

    # Lecture en BGR par défaut
    arr_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if arr_bgr is None:
        raise RuntimeError(f"OpenCV n'a pas pu lire le fichier: {path}")

    # Conversion BGR -> RGB pour cohérence
    arr = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)

    if downscale_preview > 1:
        H, W = arr.shape[:2]
        # Sous-échantillonnage simple (plus rapide que cv2.resize)
        arr = arr[::downscale_preview, ::downscale_preview]
        print(f"   Image sous-échantillonnée ({downscale_preview}x) → {arr.shape[1]}x{arr.shape[0]} px")

    # Si l'image est en niveaux de gris (ndim == 2), on la convertit en image RGB
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

    # On s'assure que le format des pixels est uint8 (valeurs 0-255)
    # cv2.imread (avec IMREAD_COLOR) lit généralement en uint8, mais
    # certains TIFF 16-bit peuvent être lus. On normalise.
    if arr.dtype != np.uint8:
        print_warning(f"Image lue en {arr.dtype}, normalisation en uint8...")
        # Normalisation robuste min-max vers 0-255
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    print_success(f"Image chargée (RGB): {arr.shape[1]}x{arr.shape[0]} px")
    return arr


# ================================================================
# DÉTECTION DU CONTOUR DE L'ÉCHANTILLON (MASQUE GLOBAL - via OpenCV)
# ================================================================

def detect_sample_contour(img, subsample_factor=10, blur_sigma=2):
    """
    Détecte le contour principal (masque binaire) d'un échantillon.
    Utilise OpenCV.
    - img : image RGB numpy
    - subsample_factor : réduction utilisée pour la détection rapide (ex: 10)
    - blur_sigma : sigma du filtre gaussien
    Retour : masque binaire de la taille originale (True = zone d'échantillon)
    """
    print_info("Détection du contour en cours...")
    start = time.time()
    H, W = img.shape[:2]

    # Sous-échantillonnage pour accélérer
    img_small = img[::subsample_factor, ::subsample_factor]

    # Conversion en gris (standard OpenCV)
    gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)

    # Application d'un flou gaussien (ksize (0,0) -> déduit de sigma)
    # sigma doit être impair s'il est utilisé pour ksize, mais pour sigmaX c'est ok
    k_size = int(blur_sigma * 4) * 2 + 1 # Estimer une taille de kernel
    gray_blur = cv2.GaussianBlur(gray, (k_size, k_size), sigmaX=blur_sigma)

    # Calcul du seuil d'Otsu (on veut le tissu, qui est sombre, donc THRESH_BINARY_INV)
    thresh_val, mask_small = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Recherche de la plus grande région connectée
    contours, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("Aucune région détectée.")

    # On choisit le contour ayant la plus grande aire
    largest_contour = max(contours, key=cv2.contourArea)

    # On crée un masque vide et on dessine le plus grand contour (rempli)
    samplemask_small = np.zeros_like(mask_small)
    cv2.drawContours(samplemask_small, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Opérations morphologiques (équivalent skimage.morphology.disk(N))
    # Note: disk(5) -> diamètre 11. disk(2) -> diamètre 5
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Fermeture (remplit petits trous)
    samplemask_small = cv2.morphologyEx(samplemask_small, cv2.MORPH_CLOSE, kernel_close)
    # Dilatation (lisse les bords)
    samplemask_small = cv2.dilate(samplemask_small, kernel_dilate, iterations=1)

    # (L'étape binary_fill_holes de scipy est omise car cv2.drawContours(FILLED)
    # et MORPH_CLOSE ont déjà rempli la structure principale)

    # On remet le masque à la taille d'origine (interpolation NEAREST pour garder 0/255)
    samplemask_full = cv2.resize(samplemask_small, (W, H), interpolation=cv2.INTER_NEAREST)

    print_success(f"Contour détecté en {time.time() - start:.1f}s")
    
    # Retourne un masque booléen
    return samplemask_full > 128


# ================================================================
# QUANTIFICATION DES STRUCTURES (SEUILLAGE DANS HSV - via OpenCV)
# ================================================================
# Utilise cv2.cvtColor et des seuils adaptés aux plages 0-179 (H) et 0-255 (S,V)

def quantify_structures(img, samplemask, downscale_factor=10):
    """
    Quantifie Collagène, Tissu et Air utile.
    Utilise OpenCV pour la conversion HSV.
    - img : image RGB full-size
    - samplemask : masque binaire full-size (True = zone à analyser)
    - downscale_factor : facteur pour réduire la résolution pendant la quantification
    Retour : dict avec clés "Collagène (%)", "Tissu (%)", "Air utile (%)"
    """
    print_info("Quantification en cours...")
    start = time.time()

    small_img = img[::downscale_factor, ::downscale_factor]
    small_mask = samplemask[::downscale_factor, ::downscale_factor]

    # Conversion RGB -> HSV (OpenCV)
    # Plages OpenCV : H [0, 179], S [0, 255], V [0, 255]
    hsv = cv2.cvtColor(small_img, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    # Définition des masques par conditions logiques :
    # Les seuils originaux (0-1) sont convertis aux plages OpenCV

    # - Air : (V > 0.85) & (S < 0.25)
    # V > 0.85 * 255 = 217
    # S < 0.25 * 255 = 64
    mask_air = (V > 217) & (S < 64) & small_mask

    # - Collagène : (H > 0.55 & H < 0.75) & (S > 0.25) & (V > 0.3)
    # H > 0.55 * 179 = 98
    # H < 0.75 * 179 = 134
    # S > 0.25 * 255 = 64
    # V > 0.3 * 255 = 76
    mask_collagen = ((H > 98) & (H < 134)) & (S > 64) & (V > 76) & small_mask

    # - Tissu : (H < 0.05 | H > 0.9) & (S > 0.25) & (V > 0.3)
    # H < 0.05 * 179 = 9
    # H > 0.9 * 179 = 161
    # S > 64
    # V > 76
    mask_tissue = ((H < 9) | (H > 161)) & (S > 64) & (V > 76) & small_mask
    
    # Nombre total de pixels pertinents (dans le masque de l'échantillon)
    total_pixels = np.sum(small_mask)
    if total_pixels == 0:
        return {"Collagène (%)": 0, "Tissu (%)": 0, "Air utile (%)": 0}

    # Calcul des pourcentages
    collagen_pct = round(np.sum(mask_collagen) / total_pixels * 100, 2)
    tissue_pct = round(np.sum(mask_tissue) / total_pixels * 100, 2)
    air_pct = round(np.sum(mask_air) / total_pixels * 100, 2)

    results = {
        "Collagène (%)": collagen_pct,
        "Tissu (%)": tissue_pct,
        "Air utile (%)": air_pct
    }

    print_success(f"Quantification terminée en {time.time() - start:.1f}s")
    return results


# ================================================================
# VISUALISATION (sauvegarde d'images illustratives - via OpenCV)
# ================================================================
# Utilise cv2.imwrite pour sauver en PNG et cv2.putText pour ajouter les légendes

def _create_canvas_with_title(img, title_lines, title_height=60):
    """Utilitaire pour ajouter un bandeau titre à une image (style matplotlib)"""
    H, W = img.shape[:2]
    # Crée un canvas noir (H + bandeau) x W
    canvas = np.zeros((H + title_height, W, 3), dtype=np.uint8)
    # Colle l'image en bas
    canvas[title_height:, :, :] = img
    
    # Ajoute les lignes de texte
    for i, line in enumerate(title_lines):
        y_pos = 25 + i * 25 # Position verticale de la ligne
        cv2.putText(canvas, line, (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas

def visualize_contour_zone(img, samplemask, output_path, timestamp):
    """
    Sauvegarde une image montrant le contour détecté en overlay.
    Utilise OpenCV.
    """
    display_factor = 20 if min(img.shape[:2]) > 5000 else 5
    img_d = img[::display_factor, ::display_factor]
    mask_d = samplemask[::display_factor, ::display_factor]

    # Création d'un overlay visuel (numpy est efficace ici)
    overlay = img_d.copy()
    # Teinte en jaune les zones exclues (~mask_d)
    yellow_part = (0.3 * overlay[~mask_d] + 0.7 * np.array([255, 255, 0])).astype(np.uint8)
    overlay[~mask_d] = yellow_part

    # Création du canvas final avec titre
    title_lines = [
        f"Contour detecte (jaune = exclu)",
        f"{timestamp}"
    ]
    canvas = _create_canvas_with_title(overlay, title_lines)

    # Sauvegarde (OpenCV écrit en BGR)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def visualize_segmentation(img, samplemask, downscale_factor, output_path, timestamp):
    """
    Sauvegarde une image représentant la segmentation (R= tissu, V= collagène, B= air).
    Utilise OpenCV.
    """
    small_img = img[::downscale_factor, ::downscale_factor]
    small_mask = samplemask[::downscale_factor, ::downscale_factor]
    
    hsv = cv2.cvtColor(small_img, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    # Recalcul des masques (seuils OpenCV 0-179, 0-255)
    mask_air = (V > 217) & (S < 64) & small_mask
    mask_collagen = ((H > 98) & (H < 134)) & (S > 64) & (V > 76) & small_mask
    mask_tissue = ((H < 9) | (H > 161)) & (S > 64) & (V > 76) & small_mask

    # Préparation d'une image RGB vide (noire)
    seg = np.zeros_like(small_img)
    # Remplissage des canaux (logique numpy)
    seg[..., 0][mask_tissue] = 255    # canal rouge = tissu
    seg[..., 1][mask_collagen] = 255  # canal vert = collagène
    seg[..., 2][mask_air] = 255       # canal bleu = air utile

    # Création du canvas final avec titre
    title_lines = [
        "Segmentation: Rouge=Tissu | Vert=Collagene | Bleu=Air utile",
        f"{timestamp}"
    ]
    canvas = _create_canvas_with_title(seg, title_lines)

    # Sauvegarde (OpenCV écrit en BGR)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


# ================================================================
# TRAITEMENT D’UNE IMAGE (fonction orchestratrice)
# ================================================================
# (Quasiment identique, appelle juste read_image_cv)

def process_single_image(image_path, output_folder, downscale_preview, downscale_factor):
    """
    Processus complet pour une seule image.
    """
    basename = Path(image_path).stem
    print_header(f"TRAITEMENT: {basename}")

    start_total = time.time()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Lecture de l'image (via OpenCV)
    img = read_image_cv(image_path, downscale_preview=downscale_preview)

    # 2) Détection du contour (via OpenCV)
    samplemask = detect_sample_contour(img, subsample_factor=10)

    # 3) Sauvegarde image overlay (via OpenCV)
    contour_path = os.path.join(output_folder, f"{basename}_contour_{timestamp_str}.png")
    visualize_contour_zone(img, samplemask, contour_path, timestamp_str)

    # 4) Sauvegarde image de segmentation (via OpenCV)
    seg_path = os.path.join(output_folder, f"{basename}_segmentation_{timestamp_str}.png")
    visualize_segmentation(img, samplemask, downscale_factor, seg_path, timestamp_str)

    # 5) Quantification (via OpenCV-HSV)
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
# CHOIX DU MODE D'ANALYSE (interaction utilisateur)
# ================================================================
# (Identique à l'original)

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
# SÉLECTION DES IMAGES (manuel ou dossier complet)
# ================================================================
# (Identique à l'original)

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
        
        for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"):
            images.extend(Path(folder).glob(ext))
        
        images = [str(p) for p in images]
        print_success(f"{len(images)} images détectées dans le dossier.")

    return images


# ================================================================
# FONCTION MAIN (orchestration générale et interface utilisateur)
# ================================================================
# (Identique à l'original)

def main():
    print_header("ANALYSE AUTOMATISÉE D'IMAGES HISTOLOGIQUES (Version OpenCV)")
    downscale_preview, downscale_factor = choose_analysis_mode()

    # PHASE 1 : test sur une image
    print_header("PHASE 1: TEST SUR UNE IMAGE")
    test_image = input("Chemin de l'image de test (ou 'q' pour quitter): ").strip()
    if test_image.lower() == "q" or not os.path.isfile(test_image):
        if test_image.lower() != "q": print_error("Fichier non trouvé.")
        print_warning("Analyse annulée.")
        return
    
    out_dir = os.path.join(os.path.dirname(test_image), "TEST_RESULTS_CV")
    os.makedirs(out_dir, exist_ok=True)
    try:
        row = process_single_image(test_image, out_dir, downscale_preview, downscale_factor)
        print_success(f"Résultats du test : Collagène={row['Collagène (%)']}%, Tissu={row['Tissu (%)']}%, Air={row['Air utile (%)']}%")
        print_info(f"Visualisations sauvegardées dans: {out_dir}")
    except Exception as e:
        print_error(f"Erreur pendant le test: {e}")
        import traceback
        traceback.print_exc() # Affiche plus de détails en cas d'erreur
        return

    if input("Valider et passer au batch ? (o/n): ").strip().lower() != "o":
        print_warning("Analyse interrompue.")
        return

    # PHASE 2 : traitement batch
    print_header("PHASE 2: TRAITEMENT BATCH")
    images = get_image_list()
    if not images:
        print_error("Aucune image fournie.")
        return
    
    out_root = input("Dossier de sortie principal pour le batch: ").strip()
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
                "Collagène (%)": "", "Tissu (%)": "", "Air utile (%)": "",
                "Temps de calcul (s)": "",
                "Erreur": str(e)
            })

    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(out_root, "ANALYSE_COMPLETE_CV.csv")
        try:
            df.to_csv(csv_path, index=False, sep=';') # Utilisation du ; pour CSV FR
            print_success(f"CSV global enregistré: {csv_path}")
        except Exception as e:
            print_error(f"Impossible de sauver le CSV: {e}")
            print_warning("Affichage des résultats dans la console :")
            print(df)


        df_ok = df[df["Erreur"] == ""].copy() # .copy() pour éviter SettingWithCopyWarning
        if not df_ok.empty:
            print_header("RÉSUMÉ GLOBAL")
            # Conversion explicite en numérique pour les calculs
            df_ok["Collagène (%)"] = pd.to_numeric(df_ok["Collagène (%)"])
            df_ok["Tissu (%)"] = pd.to_numeric(df_ok["Tissu (%)"])
            df_ok["Air utile (%)"] = pd.to_numeric(df_ok["Air utile (%)"])
            df_ok["Temps de calcul (s)"] = pd.to_numeric(df_ok["Temps de calcul (s)"])

            print(f"→ Moyenne Collagène: {df_ok['Collagène (%)'].mean():.2f}%")
            print(f"→ Moyenne Tissu: {df_ok['Tissu (%)'].mean():.2f}%")
            print(f"→ Moyenne Air utile: {df_ok['Air utile (%)'].mean():.2f}%")
            print(f"→ Temps moyen: {df_ok['Temps de calcul (s)'].mean():.1f}s")

    print_header("FIN DU PROGRAMME")


if __name__ == "__main__":
    main()
