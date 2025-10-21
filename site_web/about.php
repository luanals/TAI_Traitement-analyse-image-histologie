<?php include 'includes/header.php'; ?>

<section class="context">
    <h2>Contexte Scientifique</h2>
    <p>Ce projet a été développé au sein de Polytech Marseille avec l'orientation du Laboratoire de Biomécanique Appliquée (LBA). Il s'inscrit dans le cadre d'une recherche fondamentale visant à décrypter l'évolution et les conséquences structurelles du <b>Syndrome de Détresse Respiratoire Aigu� (SDRA)</b>.</p>
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
	automatique des cellules sont fournis figure 1.</p>
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

<section class="perspectives">
    <h2>? Perspectives Futures</h2>
    <p>Les améliorations envisagées pour ce projet incluent :</p>
    <ul>
        <li>... </li>
        <li>... </li>
    </ul>
</section>

<details>
  <summary><strong>Références</strong></summary>
  <ul>
    <li>Vyas, S. et al. (2020). <em>Neural population dynamics underlying motor learning transfer</em>. <a href="https://doi.org/10.1038/s41593-020-0603-9" target="_blank">https://doi.org/10.1038/s41593-020-0603-9</a></li>
    <li>Kobak, D. et al. (2016). <em>Demixed principal component analysis of neural population data</em>. <a href="https://doi.org/10.7554/eLife.10989" target="_blank">https://doi.org/10.7554/eLife.10989</a></li>
  </ul>
</details>


<?php include 'includes/footer.php'; ?>