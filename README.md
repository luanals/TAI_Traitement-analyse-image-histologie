# 🧬 Analyse automatisée d’images histologiques pulmonaires – SDRA  

## 📖 Contexte  
Ce projet a été développé au sein de Polytech Marseille avec l'orientation du Laboratoire de Biomécanique Appliquée (LBA) dans le cadre d’une recherche sur le **Syndrome de Détresse Respiratoire Aiguë (SDRA)**.  
L’objectif est de mieux comprendre l’impact du SDRA sur la biomécanique et la physiologie pulmonaires à partir d’images histologiques colorées au **Trichrome de Masson**.  

## 🎯 Objectifs  
- Mettre en place un **processus automatisé d’analyse d’images** pour identifier et quantifier les principales composantes pulmonaires (collagène, tissu, air).  
- Fournir une **évaluation quantitative fiable** des proportions relatives de ces structures.  
- Exporter les résultats sous forme exploitable (`.csv`) pour des analyses statistiques ultérieures.  

## ⚙️ Fonctionnalités principales  
- Prise en charge de la coloration **Trichrome de Masson**.  
- **Segmentation automatique** du collagène, du tissu et des zones d’air intra-échantillon.  
- **Exclusion du fond externe** pour éviter les artefacts liés aux zones hors coupe.  
- **Quantification des surfaces** relatives de chaque composante.  
- **Export automatique** des résultats au format `.csv`.  
- **Affichage visuel** facultatif des zones segmentées pour validation.  
- Traitement **en série de plusieurs images** tout en maintenant la pleine résolution.  

## 🧠 Technologies utilisées  
- **Langage principal :** Python  
- **Bibliothèques :**  
  - `NumPy` – gestion numérique  
  - `scikit-image`, `OpenCV` – segmentation et traitement d’images  
  - `matplotlib` – visualisation  
  - `pandas` – gestion et export des données  

## 🖥️ Installation  

1. **Cloner le dépôt :**  
   ```bash
   git clone https://github.com/luanals/TAI_Traitement-analyse-image-histologie.git
   cd TAI-SDRA
   ```

2. **Créer un environnement virtuel (optionnel mais recommandé) :**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # sous Windows : venv\Scripts\activate
   ```

3. **Installer les dépendances :**  
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Utilisation  
1. Placer les images histologiques (`.tif`, `.jpg`, `.png`, etc.) dans un dossier dédié.  
2. Exécuter le script principal :  
   ```bash
   python analyse_pulmonaire.py --input /chemin/vers/images --output resultats.csv
   ```
3. Les résultats (pourcentages de collagène, tissu, air) seront enregistrés dans un fichier `.csv`.  
4. Les images segmentées peuvent être affichées pour contrôle visuel.  

## 📁 Structure du projet  
```
.
├── analyse_pulmonaire.py      # Script principal (traitement et analyse)
├── exemples/                   # Images de test
├── resultats/                  # Résultats exportés (.csv)
├── site_web/                   # Scripts pour le site
├── requirements.txt            # Dépendances Python
└── README.md
```

## 🌐 Site web du projet  
Un site web accompagnera ce dépôt, présentant :  
- La description complète du projet et son contexte scientifique,  
- Des exemples de segmentation et de résultats quantitatifs,  
- Une documentation utilisateur détaillée,  
- +?.  

🖥️ Le site est disponible à l’adresse :  
👉 [TAI-SDRA](https://luana-lopes-santiago-etu.pedaweb.univ-amu.fr/extranet/TAI-SDRA/)

## 🧩 Perspectives  
- +?

## 🧑‍💻 Auteurs  
Projet réalisé par **Alcide Demeusy** et **Luana Lopes Santiago**, étudiants en génie biomédical à **Polytech Aix-Marseille Université**.  
