<?php include 'includes/header.php'; ?>

<section class="context">
    <h2>Contexte Scientifique</h2>
    <p>Ce projet a été développé au sein de Polytech Marseille avec l'orientation du Laboratoire de Biomécanique Appliquée (LBA). Il s'inscrit dans le cadre d'une recherche fondamentale visant à décrypter l'évolution et les conséquences structurelles du <b>Syndrome de Détresse Respiratoire Aiguë (SDRA)</b>.</p>
    <p>L'analyse repose sur l'iterprétation des images histologiques colorées spécifiquement par le <b>Trichrome de Masson</b>, qui permet de différencier clairement les fibres de collagéne (bleu) du tissu pulmonaire (rouge/magenta) et des espaces aériens.</p>
    <p>Ce travail s'inscrit dans un projet visant à étudier l'effet d'une pathologie, le Syndrôme de Détresse Respiratoire Aigu
	(SDRA) sur la biomécanique et la physiologie pulmonaire. Cette maladie, qui concerne prés de 30% des patients en
	réanimation, a un taux de mortalité trés élevé (30-40%). Les anesthésistes-réanimateurs qui prennent en charge ces
	patients rencontrent des difficultés pour leur administrer la meilleure ventilation mécanique pour maximiser leur
	chance de survie. En effet, cette pathologie est trés patient-spécifique et reste largement méconnue. Afin de mieux
	comprendre cette pathologie et ses effets sur la biomécanique pulmonaire, des tests ont été effectués pour étudier
	le comportement mécanique du tissu atteint par le SDRA. En paralléle, des échantillons de tissu ont été prélevés pour
	effectuer une analyse histologique pour éventuellement relier le comportement mécanique à la microstructure
	(cellules et matrice extra-cellulaire). Un exemple d'image histologique ainsi qu'une illustration de la segmentation
	automatique des cellules sont fournis dans <a href="examples.php" class="secondary-button">la page d'Exemples</a>.</p>
    <figure>
</section>

<hr>

<section class="authors">
    <h2>Auteurs & Technologies</h2>
    
    <h3>Réalisé par :</h3>
    <p>Projet réalisé par Alcide Demeusy et Luana Lopes Santiago, étudiants en génie biomédical à Polytech Aix-Marseille Université.</p>

    <h3>Technologies Utilisées :</h3>
    <p>Le coeur de l'analyse est développé en Python en utilisant les bibliothéques suivantes :</p>
    <ul>
        <li>`NumPy` : Gestion numérique.</li>
        <li>`scikit-image`, `OpenCV` : Segmentation et traitement d'images.</li>
        <li>`matplotlib` : Visualisation facultative.</li>
        <li>`pandas` : Gestion et export des données (`.csv`).</li>
    </ul>
</section>

<hr>

<section class="programme">
    <h2>Fonctionnalités principales du programme avec scikit-image</h2>
    <ol>
        <li>Lecture du fichier TIFF (read_tiff) :</li>
    		<ul>
        		<li>Fonction : Charge une image TIFF et la convertit en tableau RGB uint8. </li>
        		<li>Avantages : Gère les images en niveaux de gris et les images avec plus de 3 canaux. </li>
    		</ul>
        <li>Détection du contour externe (detect_sample_contour) :</li>
			<ul>
        		<li>Fonction : Détecte le contour principal de l'échantillon dans l'image. </li>
        		<li>Processus : </li>
                	<ul>
        				<li>Sous-échantillonnage de l'image pour accélérer le traitement. </li>
        				<li>Conversion en niveaux de gris et application d'un flou gaussien. </li>
                		<li>Seuillage automatique (méthode d'Otsu) pour identifier les régions sombres (tissus).</li>
                        <li>Extraction du plus grand objet et nettoyage morphologique pour obtenir un masque précis.<li>
    				</ul>
                <li>Résultat : Retourne un masque binaire indiquant la zone à analyser.</li>
    		</ul>
		<li>Quantification des structures (quantify_structures) :</li>
        	<ul>
        		<li>Fonction : Quantifie les proportions de collagène, tissu normal et air utile dans la zone détectée.</li>
        		<li>Processus : </li>
                	<ul>
        				<li>Sous-échantillonnage de l'image avec un facteur de 5 pour accélérer le traitement.</li>
        				<li>Conversion en espace de couleur HSV pour une meilleure détection des couleurs. </li>
                        <li>Application de critères de couleur pour identifier le collagène (bleu), le tissu (rouge/rose) et l'air (blanc).</li>
    				</ul>
                <li>Résultat : Retourne les pourcentages de collagène, tissu et air utile.</li>
    		</ul>
		<li>Pipeline principal (main) :</li>  
        	<ul>
        		<li>Fonction : Orchestre les étapes de traitement et génère un fichier CSV avec les résultats.</li>
        		<li>Processus : </li>
                	<ul>
        				<li>Lecture de l'image TIFF.</li>
        				<li>Détection du contour de l'échantillon. </li>
                        <li>Quantification des structures.</li>
                        <li>Export des résultats dans un fichier CSV avec des métadonnées (durée du traitement, taille de l'image, etc.).</li>
    				</ul>
                <li></li>
    		</ul>
		<li>Interface utilisateur (user_interface) :</li>
			<ul>
        		<li>Fonction : Facilite l'utilisation du script pour les utilisateurs non techniques.</li>
        		<li>Processus :</li>
                	<ul>
        				<li>Demande à l'utilisateur le chemin de l'image TIFF et le dossier de sortie.</li>
        				<li>Affiche des messages clairs pour guider l'utilisateur.</li>
    				</ul>
    		</ul>              
    </ol>
	<ul>
    	<li>Fichier de sortie</li>
        	<ul>
        		<li>Fichier CSV : Contient les résultats de l'analyse avec les métadonnées suivantes :</li>
                	<ul>
        				<li>Nom et chemin du fichier.</li>
        				<li>Date et heure de début et de fin du traitement.</li>
                        <li>Durée du traitement.</li>
                        <li>Pourcentages de collagène, tissu et air utile.</li>
                        <li>Facteur de sous-échantillonnage utilisé.</li>
                        <li>Tailles originale et réduite de l'image.</li>
    				</ul>
    		</ul>
    </ul>                

</section>
        
<section class="perspectives">
    <h2>Perspectives Futures</h2>
    <p>Les améliorations envisagées pour ce projet incluent :</p>
    <ul>
        <li>... </li>
        <li>... </li>
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
    <li>Swain, M. et al. (2018). <em>A Python (Open CV) Based Automatic Tool for Parasitemia Calcuation in Peripheral Blood Smear</em>. <a href="https://doi.org/10.1109/ICICS.2018.00096" target="_blank">https://doi.org/10.1109/ICICS.2018.00096</a></li>  </ul>
</details>


<?php include 'includes/footer.php'; ?>
