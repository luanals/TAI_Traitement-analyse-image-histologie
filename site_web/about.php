<?php 
	$page_title = "À Propos du Projet | TAI-SDRA";	
	include 'includes/header.php'; 
?>

<section class="authors">
    <h2>Auteurs & Technologies</h2>
    
    <h3>Réalisé par :</h3>
    <p>Projet réalisé par Alcide Demeusy et Luana Lopes Santiago, étudiants en génie biomédical à Polytech Aix-Marseille Université.</p>

    <h3>Technologies Utilisées :</h3>
    <p>Le cœur de l'analyse est développé en Python en utilisant les bibliothèques suivantes :</p>
    <ul>
        <li>`NumPy` : Gestion numérique.</li>
        <li>`scikit-image`, `OpenCV` : Segmentation et traitement d'images.</li>
        <li>`matplotlib` : Visualisation facultative.</li>
        <li>`pandas` : Gestion et export des données (`.csv`).</li>
    </ul>
</section>

<hr>

<section class="programme">
    <h2>Fonctionnalités principales du programme</h2>
    <ol>
        <li>Lecture du fichier TIFF (read_tiff) :
            <ul>
                <li>Fonction : Charge une image TIFF et la convertit en tableau RGB uint8.</li>
                <li>Avantages : Gère les images en niveaux de gris et les images avec plus de 3 canaux.</li>
            </ul>
        </li>

        <li>Détection du contour externe (detect_sample_contour) :
            <ul>
                <li>Fonction : Détecte le contour principal de l'échantillon dans l'image.</li>
                <li>Processus :
                    <ul>
                        <li>Sous-échantillonnage de l'image pour accélérer le traitement.</li>
                        <li>Conversion en niveaux de gris et application d'un flou gaussien.</li>
                        <li>Seuillage automatique (méthode d'Otsu) pour identifier les régions sombres (tissus).</li>
                        <li>Extraction du plus grand objet et nettoyage morphologique pour obtenir un masque précis.</li>
                    </ul>
                </li>
                <li>Résultat : Retourne un masque binaire indiquant la zone à analyser.</li>
            </ul>
        </li>

        <li>Quantification des structures (quantify_structures) :
            <ul>
                <li>Fonction : Quantifie les proportions de collagène, tissu normal et air utile dans la zone détectée.</li>
                <li>Processus :
                    <ul>
                        <li>Sous-échantillonnage de l'image avec un facteur de 5 pour accélérer le traitement.</li>
                        <li>Conversion en espace de couleur HSV pour une meilleure détection des couleurs.</li>
                        <li>Application de critères de couleur pour identifier le collagène (bleu), le tissu (rouge/rose) et l'air (blanc).</li>
                    </ul>
                </li>
                <li>Résultat : Retourne les pourcentages de collagène, tissu et air utile.</li>
            </ul>
        </li>

        <li>Interface utilisateur (user_interface) :
            <ul>
                <li>Fonction : Facilite l'utilisation du script pour les utilisateurs non techniques.</li>
                <li>Processus :
                    <ul>
                        <li>Demande à l'utilisateur le chemin de l'image TIFF et le dossier de sortie.</li>
                        <li>Affiche des messages clairs pour guider l'utilisateur.</li>
                    </ul>
                </li>
            </ul>
        </li>
    </ol>

    <ul>
        <li>Fichier de sortie
            <ul>
                <li>Fichier CSV : Contient les résultats de l'analyse avec les métadonnées suivantes :
                    <ul>
                        <li>Nom et chemin du fichier.</li>
                        <li>Date et heure de début et de fin du traitement.</li>
                        <li>Durée du traitement.</li>
                        <li>Pourcentages de collagène, tissu et air utile.</li>
                        <li>Facteur de sous-échantillonnage utilisé.</li>
                        <li>Tailles originale et réduite de l'image.</li>
                    </ul>
                </li>
            </ul>
        </li>
    </ul>
</section>
        
<section class="perspectives">
    <h2>Perspectives Futures</h2>
    <p>Les points à améliorer ou à tester dans le cadre du projet peuvent comprendre :</p>
    <ul>
        <li>
            <strong>Évolution vers un Modèle de Mélange Gaussien (GMM) :</strong> 
            Remplacer le K-Means actuel par un GMM.
            <br> Le K-Means suppose que les groupes de couleurs sont sphériques et de taille similaire, ce qui n'est pas toujours le cas en histologie. Le GMM permettrait de modéliser des formes elliptiques et d'introduire une <strong>segmentation probabiliste</strong> (Soft Clustering), gérant mieux les nuances subtiles là où le K-Means impose une coupure trop stricte.
        </li>

        <li>
            <strong>Enrichissement des données (Clustering 7D) :</strong> 
            Améliorer l'espace de caractéristiques utilisé pour le clustering. Au lieu de se limiter aux 3 canaux HSV, l'objectif est d'alimenter l'algorithme avec <strong>7 dimensions par pixel</strong> : RGB (3) + Intensité moyenne (1) + Composantes HSV (3).
            <br> En combinant la colorimétrie (RGB/HSV) et la luminance (Intensité), nous donnerions à l'algorithme plus d'indices pour différencier des structures biologiquement différentes mais visuellement proches.
        </li>
        
        <li>
            <strong>Exploration de bibliothèques spécialisées (ex: HistomicsTK) :</strong> 
            Intégrer des outils de <strong>normalisation de coloration</strong> en amont de notre segmentation. Cela permettrait de corriger les variations d'exposition ou de vieillissement des colorants chimiques entre différents lots d'images, rendant notre clustering (K-Means ou GMM) beaucoup plus robuste.
        </li>
    </ul>
</section>

<details>
  <summary><strong>Références</strong></summary>
  <ul>
    <li>Kosaraju, S. C. et al. (2022). <em>Deep learning-based framework for slide-based histopathological image analysis</em>. <a href="https://doi.org/10.1038/s41598-022-23166-0" target="_blank">https://doi.org/10.1038/s41598-022-23166-0</a></li>
    <li>Pourakpour, F. et al. (2025). <em>HistomicsTK: A Python toolkit for pathology image analysis algorithms</em>. <a href="https://www.sciencedirect.com/science/article/pii/S2352711025002845" target="_blank">https://www.sciencedirect.com/science/article/pii/S2352711025002845</a></li>
    <li>van der Walt, S. et al. (2014). <em>scikit-image: image processing in Python</em>. <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4081273/" target="_blank">https://pmc.ncbi.nlm.nih.gov/articles/PMC4081273/</a></li>
    <li>Asyahira, R. & Hakiki, R. (2021). <em>The utilization openCV to measure the ammonia and color concentration in the water</em>. <a href="https://e-journal.president.ac.id/index.php/JENV/article/view/1475" target="_blank">https://e-journal.president.ac.id/index.php/JENV/article/view/1475</a></li>
    <li>Wright, A. et al. (2023). <em>Free and open-source software for object detection, size, and colour determination for use in plant phenotyping</em>. <a href="https://doi.org/10.1186/s13007-023-01103-0" target="_blank">https://doi.org/10.1186/s13007-023-01103-0</a></li>
    <li>Ing, G. et al. (2023). <em>SimpliPyTEM: An open-source Python library and app to simplify Transmission Electron Microscopy and in situ-TEM image analysis</em>. <a href="https://doi.org/10.1101/2023.04.28.538777" target="_blank">https://doi.org/10.1101/2023.04.28.538777</a></li>
    <li>Gupta, A. et al. (2025). <em>Predicting Renal Cell Carcinoma Subtypes and Fuhrman Grading Using Multiphasic CT-Based Texture Analysis and Machine Learning Techniques</em>. <a href="https://doi.org/10.1055/s-0044-1796639" target="_blank">https://doi.org/10.1055/s-0044-1796639</a></li>
    <li>Lazko, F. F. (2020). <em>Overview and Comparison of Python Image Processing Tools with Gabor Functions</em>. <a href="https://doi.org/10.32523/2616-7182/2020-132-3-25-30" target="_blank">https://doi.org/10.32523/2616-7182/2020-132-3-25-30</a></li>
    <li>Swain, M. et al. (2018). <em>A Python (Open CV) Based Automatic Tool for Parasitemia Calculation in Peripheral Blood Smear</em>. <a href="https://doi.org/10.1109/ICICS.2018.00096" target="_blank">https://doi.org/10.1109/ICICS.2018.00096</a></li>  
	<li>Thille, A. W. et al. (2013). <em>Chronology of histological lesions in acute respiratory distress syndrome</em>. <a href="https://www.thelancet.com/journals/lanres/article/PIIS2213-2600(13)70053-5/abstract" target="_blank">https://www.thelancet.com/journals/lanres/article/PIIS2213-2600(13)70053-5/abstract</a></li>
    <li>Seger, S. et al. (2018). <em>A fully automated image analysis method to quantify lung fibrosis in the bleomycin mouse model</em>. <a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0193057" target="_blank">https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0193057</a></li>
    <li>Van Heest, A. et al. (2025). <em>Quantitative Assessment of Pulmonary Fibrosis in a Murine Model Using Masson's Trichrome Staining</em>. <a href="https://pubs.acs.org/doi/10.1021/acsomega.4c08091" target="_blank">https://pubs.acs.org/doi/10.1021/acsomega.4c08091</a></li>
    <li>Ma, J. et al. (2024). <em>Segment anything in medical images</em>. <a href="https://www.nature.com/articles/s41467-024-44824-z" target="_blank">https://www.nature.com/articles/s41467-024-44824-z</a></li>
    <li>Li, S. et al. (2025). <em>Artificial intelligence and machine learning in acute respiratory distress syndrome</em>. <a href="https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1597556/full" target="_blank">https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1597556/full</a></li>
    <li>Hema, D. et al. (2019). <em>Interactive Color Image Segmentation using HSV Color Space</em>. <a href="https://www.researchgate.net/publication/341857676_Interactive_Color_Image_Segmentation_using_HSV_Color_Space" target="_blank">https://www.researchgate.net/publication/341857676_Interactive_Color_Image_Segmentation_using_HSV_Color_Space</a></li>
    <li>Zheng, J. et al. (2024). <em>Study on lung CT image segmentation algorithm based on Otsu</em>. <a href="https://www.nature.com/articles/s41598-024-68721-3" target="_blank">https://www.nature.com/articles/s41598-024-68721-3</a></li>
    <li>Editverse. (2025). <em>10 Essential Scikit-Image Techniques for Medical Image Analysis</em>. <a href="https://editverse.com/10-essential-scikit-image-techniques-for-medical-image-analysis/" target="_blank">https://editverse.com/10-essential-scikit-image-techniques-for-medical-image-analysis/</a></li>
  </ul>
	
</details>


<?php include 'includes/footer.php'; ?>