"""
TAI.py — Version commentée sans detection de bordes

But :
    Découper une grande image TIFF en tuiles, extraire le pourcentage de collagène
    (coloration bleue typique du Trichrome de Masson) et le pourcentage d'air
    (fond blanc), puis écrire un fichier CSV avec une ligne par tuile et un total.

Contexte :
    Outil destiné à l'analyse automatisée d'images histologiques pulmonaires (Trichrome)
    dans le cadre d'études sur le SDRA. Conçu pour traiter des images de très haute
    résolution (mégapixels / gigapixels) en découpant l'image en tuiles et en
    appliquant des heuristiques de segmentation couleur + méthodes de secours
    (KMeans, déconvolution) si nécessaire.

Dépendances recommandées :
    - numpy
    - tifffile (fortement recommandé pour gros TIFF)
    - imageio (fallback pour lecture TIFF)
    - scikit-image (conversion d'espace couleur, seuillage Otsu)
    - scikit-learn (KMeans si clustering requis)
    - pandas (facultatif pour export CSV, on utilise csv standard ici)
    - matplotlib (pour visualisation interactive)

Remarques générales :
    - 

Perspectives:
    -    

Exemple d'utilisation rapide :
    from TAI_commented_fr import run_pipeline
    run_pipeline('sample.tif', out_csv='resultats.csv', nx=8, ny=8)

"""

import os
import math
import csv
import numpy as np
from datetime import datetime

# Imports optionnels : on tente d'importer des bibliothèques utiles, mais
# on laisse le module fonctionner avec des fonctionnalités réduites si
# certaines bibliothèques ne sont pas disponibles.
try:
    import tifffile as tiff  # lecture TIFF efficace (préféré pour grandes images)
except Exception:
    tiff = None
try:
    from skimage import color, filters, exposure
except Exception:
    # skimage facultatif — certaines utilités comme rgb2lab, threshold_otsu
    # seront indisponibles si skimage n'est pas installé.
    color = None
    filters = None
    exposure = None
try:
    from sklearn.cluster import KMeans
except Exception:
    # KMeans n'est qu'un recours pour séparer visuellement clusters couleur.
    KMeans = None
try:
    import imageio  # fallback pour lecture/écriture d'images
except Exception:
    imageio = None

# -----------------------
# 1) Lecture du TIFF
# -----------------------
# Objectif : lire un fichier TIFF en mémoire en renvoyant un tableau numpy
#           de type uint8 et de forme (H, W, 3) (canaux RGB).
# Contraintes :
#    - Les images peuvent être en niveaux de gris ou posséder >3 canaux.
#    - Les valeurs de type peuvent varier (float32, uint16...). On normalise
#      vers uint8 pour homogénéité des traitements suivants.
# Retour : numpy.ndarray dtype=uint8 shape (H,W,3)

def read_tiff(path: str):
    """Lit un TIFF et renvoie un array RGB uint8.

    Comportement détaillé :
      - Utilise tifffile.imread si disponible (meilleur support des gros TIFF).
      - Si la lecture échoue ou que tifffile n'est pas présent, utilise imageio.
      - Si l'image est en niveaux de gris (ndim == 2), elle est dupliquée
        sur 3 canaux pour obtenir un image RGB.
      - Si l'image possède plus de 3 canaux, on ne conserve que les 3 premiers
        (souvent correspondant à RGB ou à des canaux utiles).
      - Si le dtype n'est pas uint8, on remet à l'échelle les intensités vers
        0..255 et on cast en uint8. Si scikit-image (exposure) est présent,
        on utilise rescale_intensity pour un résultat plus robuste.

    Paramètres :
      path (str) : chemin du fichier TIFF.

    Retour :
      arr (np.ndarray) : image RGB uint8.
    """
    # Tentative avec tifffile (si présent)
    if tiff is not None:
        try:
            arr = tiff.imread(path)
        except Exception:
            arr = None
    else:
        arr = None

    # Fallback : imageio
    if arr is None:
        if imageio is None:
            raise RuntimeError("Besoin de tifffile ou imageio pour lire TIFF.")
        arr = imageio.imread(path)

    # Si grayscale -> dupliquer canaux pour obtenir RGB
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    # Si >3 canaux -> conserver les 3 premiers (sécurité)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]

    # Normaliser le type vers uint8
    if arr.dtype != np.uint8:
        if exposure is not None:
            # scikit-image offre une normalisation robuste
            arr = exposure.rescale_intensity(arr, out_range='uint8').astype(np.uint8)
        else:
            # fallback manuel : min-max scaling
            arr = ((arr - arr.min()) / (arr.ptp() + 1e-9) * 255).astype(np.uint8)
    return arr


# -----------------------
# 2) Découpage en tuiles
# -----------------------
# Objectif : itérer sur des régions (tuiles) couvrant l'image entière.
# Contrainte : permettre un recouvrement (overlap) paramétrable entre tuiles
#              pour éviter les artefacts de bord ou capturer des structures
#              qui chevauchent plusieurs tuiles.
# Design :
#    - Calcul des dimensions de tuiles en se basant sur nx, ny (nombre de
#      tuiles souhaité approximativement par axe).
#    - Le pas (step) est ajusté selon le recouvrement.
#    - On s'assure que la dernière tuile touche le bord droit/bas de l'image.
# Format de sortie : générateur qui renvoie des tuples
#    ((xi, yi), (x0, y0, x1, y1), tile_array)


def tile_image(img: np.ndarray, nx: int, ny: int, overlap: float = 0.0):
    """Génère des tuiles approximativement nx x ny.

    Paramètres :
      img : tableau numpy HxWx3
      nx, ny : nombre approximatif de tuiles sur l'axe x et y
      overlap : fraction (0.0 - 0.5) correspond au recouvrement entre tuiles

    Yields :
      ( (xi, yi), (x0, y0, x1, y1), tile_array )
      - xi, yi : indices de tuile
      - (x0,y0,x1,y1) : coordonnées de la tuile dans l'image originale
      - tile_array : copie numpy de la tuile (HxWx3)

    Remarques :
      - La fonction retourne des copies des régions pour éviter les problèmes
        liés aux vues sur le tableau original quand on modifie les tuiles.
      - Si overlap est 0, les tuiles sont contiguës ; valeurs plus élevées
        augmentent la redondance et le temps de calcul.
    """
    H, W = img.shape[:2]
    tile_w = math.ceil(W / nx)  # largeur approximative d'une tuile
    tile_h = math.ceil(H / ny)  # hauteur approximative d'une tuile

    # Le pas est la portion non-recouverte (au moins 1 pixel)
    step_x = int(tile_w * (1.0 - overlap)) or 1
    step_y = int(tile_h * (1.0 - overlap)) or 1

    x_starts = list(range(0, W, step_x))
    y_starts = list(range(0, H, step_y))

    # S'assurer que la dernière tuile touche le bord (évite petites marges non couverts)
    if x_starts and x_starts[-1] + tile_w < W:
        x_starts[-1] = max(0, W - tile_w)
    if y_starts and y_starts[-1] + tile_h < H:
        y_starts[-1] = max(0, H - tile_h)

    for xi, x0 in enumerate(x_starts):
        for yi, y0 in enumerate(y_starts):
            x1 = min(W, x0 + tile_w)
            y1 = min(H, y0 + tile_h)
            # On renvoie une copie afin que l'appelant puisse modifier la tuile
            yield (xi, yi), (x0, y0, x1, y1), img[y0:y1, x0:x1].copy()


# -----------------------
# 2bis) Visualisation d'une tuile par coordonnées
# -----------------------
# Utilité : fonction interactive pour inspection manuelle d'une tuile donnée.
#          Fournit affichages et mesures pour valider heuristiques de seuils.
# Attention : dépend de matplotlib — usage en environnement interactif / notebook.


def visualize_tile_by_coords(image_path: str, x0, y0, x1, y1):
    """Charge l'image, extrait la tuile spécifiée et affiche :
       - l'image originale
       - le masque collagène (bleu) selon heuristiques
       - le masque background (air/blanc)
       - le masque "tissu" (reste)

    Cette fonction est destinée à la validation visuelle et réglage de
    paramètres (seuils)."""
    import matplotlib.pyplot as plt

    print(f"Lecture image...")
    img = read_tiff(image_path)

    print(f"Extraction tuile [{x0}:{x1}, {y0}:{y1}]...")
    tile = img[y0:y1, x0:x1]

    H, W = tile.shape[:2]
    print(f"Taille tuile: {W}x{H} pixels")

    # Séparer canaux en float32 pour calculs
    R = tile[..., 0].astype(np.float32)
    G = tile[..., 1].astype(np.float32)
    B = tile[..., 2].astype(np.float32)

    # Heuristiques collagène (paramètres empiriques basés sur Trichrome)
    blue_very_dominant = (B > R + 40) & (B > G + 30)
    blue_intense = B > 150
    blue_ratio = B / (R + G + B + 1e-6)
    blue_ratio_high = blue_ratio > 0.45
    not_too_bright = (R + G + B) < 650
    not_too_dark = (R + G + B) > 150
    coll_mask = blue_very_dominant & blue_intense & blue_ratio_high & not_too_bright & not_too_dark

    # Heuristiques background (air/blanc) : pixels très lumineux avec faible variance
    brightness = (R + G + B) / 3.0
    very_bright = brightness > 200
    color_variance = np.std(np.stack([R, G, B], axis=-1), axis=-1)
    balanced_color = color_variance < 30
    bg_mask = very_bright & balanced_color

    pct_coll = coll_mask.sum() / (H * W) * 100.0
    pct_bg = bg_mask.sum() / (H * W) * 100.0
    pct_tissue = 100.0 - pct_coll - pct_bg

    print(f"\n=== RÉSULTATS CORRIGÉS ===")
    print(f"Collagène (bleu): {pct_coll:.2f}%")
    print(f"Air/Background (blanc): {pct_bg:.2f}%")
    print(f"Tissu normal (reste): {pct_tissue:.2f}%")
    print(f"TOTAL: {pct_coll + pct_bg + pct_tissue:.2f}%")

    # Affichage visuel (4 panneaux)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    axes[0].imshow(tile)
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(coll_mask, cmap='Blues')
    axes[1].set_title(f'Collagène: {pct_coll:.2f}%', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(bg_mask, cmap='Greys')
    axes[2].set_title(f'Air: {pct_bg:.2f}%', fontsize=14)
    axes[2].axis('off')

    tissue_mask = ~(coll_mask | bg_mask)
    axes[3].imshow(tissue_mask, cmap='Reds')
    axes[3].set_title(f'Tissu: {pct_tissue:.2f}%', fontsize=14)
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()


# -----------------------
# 3) Utilitaires couleur / OD (Densité Optique)
# -----------------------
# Context : la déconvolution de Ruifrok requiert souvent de travailler en
# espace OD : OD = -ln(I/I0). Ici I0 est 255 (uint8) et nous utilisons
# OD pour linéariser la relation entre concentration de colorant et
# intensité mesurée.


def rgb_to_od(I):
    """Convertit une image RGB uint8 en densité optique.

    Détails :
      - On convertit en float32 pour calculs.
      - On évite log(0) en remplaçant les zéros par 1.

    Entrée : I dtype uint8
    Sortie : OD float32 (mêmes dimensions que I)
    """
    I = I.astype(np.float32)
    I[I == 0] = 1.0  # éviter log(0)
    OD = -np.log(I / 255.0)
    return OD


def od_to_rgb(OD):
    """Inverse OD -> RGB uint8.

    Utilisé pour vérifier des reconstructions après déconvolution.
    """
    I = (np.exp(-OD) * 255.0).clip(0, 255).astype(np.uint8)
    return I


# -----------------------
# 4) Déconvolution (Ruifrok style) — optionnel
# -----------------------
# Objectif : si l'on connaît (ou estime) la matrice de teintes (stain_matrix)
# alors on peut projeter l'image en espace OD sur les vecteurs de teinte et
# obtenir des cartes de "concentration" par tache (ex : collagène, cytoplasme,
# hématoxyline...). Cette méthode est souvent plus robuste qu'un simple seuillage
# RGB pour des images colorées de façon standard.


def deconvolve_ruifrok(tile_rgb: np.ndarray, stain_matrix: np.ndarray):
    """Effectue une déconvolution couleur selon une matrice de teintes fournie.

    Paramètres :
      tile_rgb : HxWx3 uint8
      stain_matrix : shape (3, n_stains) ou (3,3) ; colonnes = vecteurs OD

    Retour :
      conc_img : HxWx(n_stains) float32, cartes de concentration (esp. linéaire)

    Remarques :
      - Utilise pseudoinverse pour résoudre le système (stable si matrice
        non carrée ou mal conditionnée).
      - Nécessite que la matrice de teintes provienne d'une calibration ou
        d'une estimation (p.ex. méthodes automatiques existantes pour
        estimer les vecteurs de teinte depuis l'image).
    """
    OD = rgb_to_od(tile_rgb)  # HxWx3
    H, W, _ = OD.shape
    resh = OD.reshape(-1, 3)  # (Npixels, 3)
    M = np.array(stain_matrix, dtype=np.float32)
    # Résolution linéaire via pseudoinverse : conc = resh @ pinv(M)
    pinv = np.linalg.pinv(M)
    conc = np.dot(resh, pinv.T)  # (Npixels, n_stains)
    conc_img = conc.reshape(H, W, -1)
    return conc_img


# -----------------------
# 5) Séparation via KMeans (fallback ou méthode principale)
# -----------------------
# But : utiliser un clustering non-supervisé pour séparer zones de couleur
#      différentes. Utile lorsque la déconvolution n'est pas possible ou
#      lorsque l'on souhaite obtenir des masques rapides par similarité couleur.
# Entrées : tile_rgb et paramètres de clustering.
# Sorties : labels (HxW), index du cluster collagène, index cluster background,
#           centres_rgb (estimation des couleurs moyennes par cluster)


def cluster_deconvolution(tile_rgb: np.ndarray, n_clusters: int = 3, space='lab'):
    """Sépare l'image en n_clusters à l'aide de KMeans sur un espace couleur.

    Paramètres :
      tile_rgb : HxWx3 uint8
      n_clusters : entier, nombre de clusters demandés
      space : 'lab'|'hsv'|'rgb' — espace couleur utilisé pour le clustering

    Renvoie :
      labels : HxW int (étiquette cluster par pixel)
      collagen_idx : index du cluster choisi comme collagène (max composante bleue)
      bg_idx : index du cluster correspondant au fond (max luminosité)
      centers_rgb : tableau (n_clusters, 3) des couleurs moyennes en RGB (0..255)

    Remarques :
      - Sous-échantillonne X si trop de pixels pour limiter la mémoire lors du fit.
      - KMeans nécessite scikit-learn : si absent, la fonction lève une erreur.
      - La sélection des clusters collagène/bg est heuristique et doit être
        validée sur vos propres données.
    """
    H, W = tile_rgb.shape[:2]
    # Conversion d'espace couleur pour une séparation perceptuelle meilleure
    if space == 'lab' and color is not None:
        arr = color.rgb2lab(tile_rgb)
    elif space == 'hsv' and color is not None:
        arr = color.rgb2hsv(tile_rgb)
    else:
        arr = tile_rgb / 255.0

    X = arr.reshape(-1, arr.shape[-1])

    # Sous-échantillonnage pour limiter la charge mémoire si très grand
    sample_X = X
    if X.shape[0] > 200000:
        idx = np.random.choice(X.shape[0], size=200000, replace=False)
        sample_X = X[idx]

    if KMeans is None:
        raise RuntimeError("scikit-learn manquant: installer scikit-learn pour KMeans.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=4).fit(sample_X)
    labels = kmeans.predict(X).reshape(H, W)
    centers = kmeans.cluster_centers_

    # Reprojeter centres vers RGB pour interprétation humaine
    if space == 'lab' and color is not None:
        centers_rgb = []
        for c in centers:
            lab = np.array(c, dtype=np.float32).reshape(1, 1, 3)
            rgb = (color.lab2rgb(lab)[0, 0] * 255.0)
            centers_rgb.append(rgb)
        centers_rgb = np.array(centers_rgb)
    elif space == 'hsv' and color is not None:
        centers_rgb = []
        for c in centers:
            hsv = np.array(c, dtype=np.float32).reshape(1, 1, 3)
            rgb = (color.hsv2rgb(hsv)[0, 0] * 255.0)
            centers_rgb.append(rgb)
        centers_rgb = np.array(centers_rgb)
    else:
        centers_rgb = (centers * 255.0)

    # Heuristique : cluster collagène = cluster avec la composante bleue moyenne la plus élevée
    blue_vals = centers_rgb[:, 2]
    collagen_idx = int(np.argmax(blue_vals))

    # Heuristique : cluster background = cluster le plus lumineux (somme R+G+B)
    brightness = centers_rgb.sum(axis=1)
    bg_idx = int(np.argmax(brightness))

    return labels, collagen_idx, bg_idx, centers_rgb


# -----------------------
# 6) Seuillage helper
# -----------------------
# But : fournir une fonction utilitaire qui renvoie un masque binaire et le
# seuil utilisé. Permet d'utiliser Otsu si disponible, ou percentiles/manual.


def threshold_channel(channel: np.ndarray, method='otsu', manual_thresh: float = None):
    """Seuillage d'un canal (ou d'une carte) renvoyant mask bool et seuil.

    Paramètres :
      channel : array (0..255) ou float
      method : 'otsu'|'pXX' (percentile)|'median' etc.
      manual_thresh : si spécifié, utilisé tel quel (valeur normalisée 0..1)

    Retour :
      mask (bool array), t (valeur seuil 0..1 normalisée)
    """
    c = channel.astype(np.float32)
    # normaliser 0..1 (robuste même si channel est déjà 0..1)
    c = (c - c.min()) / (c.ptp() + 1e-9)
    if manual_thresh is not None:
        t = float(manual_thresh)
    else:
        if method == 'otsu' and filters is not None:
            t = filters.threshold_otsu((c * 255).astype(np.uint8)) / 255.0
        elif method.startswith('p'):
            # ex: 'p85' -> 85ème percentile
            p = float(method[1:]) if len(method) > 1 else 50.0
            t = np.percentile(c, p)
        else:
            # fallback median
            t = float(np.median(c))
    mask = c >= t
    return mask, t


# -----------------------
# 7) Analyse d'une tuile (orchestration)
# -----------------------
# Cette fonction regroupe les heuristiques pour détecter :
#   - collagène (bleu), via combinaison de conditions sur composantes RGB
#   - background/air (blanc), via luminosité et faible variance couleur
# Elle renvoie les pourcentages et un dict debug utile pour analyses.


def analyze_tile(tile_rgb: np.ndarray, deconv_matrix=None, manual_thresh_collagen: float = None):
    """Analyse une tuile et retourne pourcentages collagène et background.

    Paramètres :
      tile_rgb : HxWx3 uint8
      deconv_matrix : si fourni, on pourrait utiliser la déconvolution
                      (actuellement non intégrée automatiquement ici)
      manual_thresh_collagen : seuil manuel pour la composante collagène

    Retour :
      pct_coll (float) : pourcentage de surface détectée comme collagène
      pct_bg (float) : pourcentage de surface détectée comme background/air
      debug (dict) : informations additionnelles (nombres pixels, méthode...)

    Remarques :
      - Les seuils sont empiriques (adaptés à Trichrome) et doivent être
        vérifiés/ajustés sur vos données (fonction diagnose_quick aide).
      - On considère 'tissu' = reste (100 - coll - bg).
    """
    H, W = tile_rgb.shape[:2]
    total_pixels = H * W
    debug = {}

    R = tile_rgb[..., 0].astype(np.float32)
    G = tile_rgb[..., 1].astype(np.float32)
    B = tile_rgb[..., 2].astype(np.float32)

    # 1. COLLAGÈNE (bleu intense) — combinaison de critères
    blue_very_dominant = (B > R + 40) & (B > G + 30)
    blue_intense = B > 150
    total_intensity = R + G + B + 1e-6
    blue_ratio = B / total_intensity
    blue_ratio_high = blue_ratio > 0.45
    not_too_bright = (R + G + B) < 650
    not_too_dark = (R + G + B) > 150

    coll_mask = blue_very_dominant & blue_intense & blue_ratio_high & not_too_bright & not_too_dark

    # 2. BACKGROUND/AIR (zones BLANCHES)
    brightness = (R + G + B) / 3.0
    very_bright = brightness > 200  # seuil ajusté empiriquement

    # Vérifier que c'est blanc (pas juste bleu clair) : faible écarts entre R,G,B
    color_variance = np.std(np.stack([R, G, B], axis=-1), axis=-1)
    balanced_color = color_variance < 30

    bg_mask = very_bright & balanced_color

    pct_coll = coll_mask.sum() / total_pixels * 100.0
    pct_bg = bg_mask.sum() / total_pixels * 100.0
    pct_tissue = 100.0 - pct_coll - pct_bg

    debug.update({
        'method': 'corrected_white_detection',
        'blue_pixels': int(coll_mask.sum()),
        'bg_pixels': int(bg_mask.sum()),
        'tissue_pct': pct_tissue,
    })

    return pct_coll, pct_bg, debug


# -----------------------
# 7bis) Diagnostic rapide (aide au réglage)
# -----------------------
# Fonction utilitaire pour extraire quelques statistiques de la zone centrale
# de l'image afin d'ajuster seuils et paramètres (utile lors de calibrage).


def diagnose_quick(image_path: str):
    """Affiche statistiques rapides sur la zone centrale d'une image.

    Objectif : comprendre la dynamique des canaux R/G/B et repérer un pixel
               "très bleu" typique pour aider à paramétrer les heuristiques.
    """
    print("Lecture image...")
    img = read_tiff(image_path)
    H, W = img.shape[:2]

    # Extraire une zone centrale raisonnable (safe bounds)
    center_y, center_x = H // 2, W // 2
    tile = img[center_y:center_y + 1000, center_x:center_x + 1000]

    # Sous-échantillonnage pour accélérer
    tile = tile[::10, ::10]

    R = tile[..., 0].astype(np.float32)
    G = tile[..., 1].astype(np.float32)
    B = tile[..., 2].astype(np.float32)

    print("\n=== DIAGNOSTIC COULEURS (zone centrale) ===")
    print(f"Canal Rouge - min: {R.min():.0f}, max: {R.max():.0f}, moy: {R.mean():.0f}")
    print(f"Canal Vert  - min: {G.min():.0f}, max: {G.max():.0f}, moy: {G.mean():.0f}")
    print(f"Canal Bleu  - min: {B.min():.0f}, max: {B.max():.0f}, moy: {B.mean():.0f}")

    # Pixel avec le score bleu le plus élevé (heuristique simple)
    blue_score = B - (R + G) / 2
    top_idx = np.argmax(blue_score)
    y, x = np.unravel_index(top_idx, (tile.shape[0], tile.shape[1]))
    r, g, b = tile[y, x]

    print(f"\nPixel le PLUS BLEU trouvé: RGB({r}, {g}, {b})")
    print(f"Ratio bleu de ce pixel: {b / (r + g + b + 1e-6):.3f}")

    blue_ratio = B / (R + G + B + 1e-6)
    print(f"\nRatio bleu moyen: {blue_ratio.mean():.3f}")
    print(f"% pixels avec ratio bleu > 0.4: {(blue_ratio > 0.4).sum() / blue_ratio.size * 100:.2f}%")


# -----------------------
# 8) Pipeline global + CSV
# -----------------------
# Fonction principale pour traiter une image entière :
#   - lit l'image
#   - découpe en tuiles
#   - analyse chaque tuile (analyze_tile)
#   - optionnel : sauvegarde masques par tuile
#   - agrège les résultats et écrit un CSV
# Retour : liste d'enregistrements + dictionnaire agrégé


def run_pipeline(image_path: str, out_csv: str = "results_tiles.csv", nx: int = 8, ny: int = 8, overlap: float = 0.0, deconv_matrix=None, save_tile_masks: bool = False, mask_out_dir: str = "masks"):
    """Pipeline complet pour analyser une image et exporter résultats CSV.

    Paramètres :
      image_path : chemin du TIFF
      out_csv : fichier de sortie CSV
      nx, ny : nombre approximatif de tuiles sur x et y
      overlap : recouvrement fractionnaire entre tuiles
      deconv_matrix : matrice de teintes (optionnelle)
      save_tile_masks : si True, sauvegarde masques collagène/bg par tuile (png)
      mask_out_dir : répertoire où écrire les masques

    Retour :
      records : liste de dicts (une entrée par tuile)
      totals : dict avec pourcentages totaux agrégés

    Remarques :
      - Le CSV contient une ligne de total en fin de fichier (tile_x='TOTAL').
      - L'export CSV utilise le module csv natif ; pour de l'analyse poussée,
        vous pouvez remplacer par pandas.DataFrame(records).to_csv(...)
    """
    print("Reading:", image_path)
    img = read_tiff(image_path)
    H, W = img.shape[:2]
    print(f"Image size: {W} x {H}")

    records = []
    if save_tile_masks and not os.path.exists(mask_out_dir):
        os.makedirs(mask_out_dir)

    tile_count = 0
    for (xi, yi), (x0, y0, x1, y1), tile in tile_image(img, nx=nx, ny=ny, overlap=overlap):
        tile_count += 1
        pct_coll, pct_bg, debug = analyze_tile(tile, deconv_matrix)
        records.append({
            'tile_x': xi,
            'tile_y': yi,
            'x0': x0,
            'y0': y0,
            'x1': x1,
            'y1': y1,
            'pixels': tile.shape[0] * tile.shape[1],
            'pct_collagen': pct_coll,
            'pct_background': pct_bg,
            'debug': str(debug),
        })
        print(f"Tile {tile_count} ({xi},{yi}) -> coll {pct_coll:.2f}%, bg {pct_bg:.2f}%")

        # Option : sauvegarder masques visuels par tuile en utilisant le clustering
        if save_tile_masks:
            try:
                import imageio
                labels, coll_idx, bg_idx, centers = cluster_deconvolution(tile, n_clusters=3, space='lab')
                coll_mask = (labels == coll_idx).astype(np.uint8) * 255
                bg_mask = (labels == bg_idx).astype(np.uint8) * 255
                out_mask = np.stack([coll_mask, bg_mask, np.zeros_like(coll_mask)], axis=2)
                imageio.imwrite(os.path.join(mask_out_dir, f"mask_x{xi}_y{yi}.png"), out_mask)
            except Exception as e:
                print("Warning: can't save mask:", e)

    # Agrégation des totaux (moyenne pondérée par nombre de pixels)
    total_pixels = sum([r['pixels'] for r in records])
    total_coll_pixels = sum([r['pct_collagen'] * r['pixels'] / 100.0 for r in records])
    total_bg_pixels = sum([r['pct_background'] * r['pixels'] / 100.0 for r in records])
    total_coll_pct = total_coll_pixels / total_pixels * 100.0 if total_pixels > 0 else 0.0
    total_bg_pct = total_bg_pixels / total_pixels * 100.0 if total_pixels > 0 else 0.0

    # Écriture CSV
    fieldnames = ['tile_x', 'tile_y', 'x0', 'y0', 'x1', 'y1', 'pixels', 'pct_collagen', 'pct_background', 'debug']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
        writer.writerow({'tile_x': 'TOTAL', 'pixels': total_pixels, 'pct_collagen': total_coll_pct, 'pct_background': total_bg_pct, 'debug': datetime.utcnow().isoformat() + 'Z'})

    print(f"Done: wrote {len(records)} tiles to {out_csv}. Totals: collagen {total_coll_pct:.2f}% bg {total_bg_pct:.2f}%")
    return records, {'total_collagen_pct': total_coll_pct, 'total_background_pct': total_bg_pct}


# --- Exemples d'usage (décommenter/adapter) ---
# read_tiff("/chemin/vers/image.tif")
# visualize_tile_by_coords("/chemin/vers/image.tif", x0=5760, y0=25920, x1=11520, y1=34560)
# run_pipeline("/chemin/vers/image.tif")
