<?php
   
    $page_title = "Aperçu du Projet | TAI-SDRA";
    
    // Inclure le Header
    include 'includes/header.php';
?>

<section class="hero">
    <h2>Analyse automatisée d’images histologiques pulmonaires – SDRA</h2>
    <p class="lead">Un outil développé pour mieux comprendre l'impact du <b>Syndrome de Détresse Respiratoire Aiguë (SDRA)</b> sur la biomécanique pulmonaire à partir d’images colorées au <b>Trichrome de Masson</b>.</p>
    <p> Projet développé en partenariat avec le Laboratoire de Biomécanique Appliquée (LBA)</p>
</section>

<section class="objectives">
    <h3>🎯 Objectifs Principaux</h3>
    <ul>
        <li>Mise en place d'un <b>processus automatisé d’analyse</b> pour identifier et quantifier les composantes (collagène, tissu, air).</li>
        <li>Fournir une <b>évaluation quantitative fiable</b> des proportions relatives.</li>
        <li>Exporter les résultats sous forme exploitable (`.csv`) pour des analyses statistiques.</li>
    </ul>
</section>

<section class="context">
    <h2>Contexte Scientifique</h2>
    <p>Ce projet a été développé au sein de Polytech Marseille avec l'orientation du Laboratoire de Biomécanique Appliquée (LBA). Il s'inscrit dans le cadre d'une recherche fondamentale visant à décrypter l'évolution et les conséquences structurelles du <b>Syndrome de Détresse Respiratoire Aiguë (SDRA)</b>.</p>
    <p>L'analyse repose sur l'interprétation d'images histologiques colorées spécifiquement par le <b>trichrome de Masson</b>, qui permet de différencier clairement les fibres de collagène (bleu) du tissu pulmonaire (rouge/magenta) et des espaces aériens.</p>
    <p>Ce travail s'inscrit dans un projet visant à étudier l'effet d'une pathologie, le Syndrome de Détresse Respiratoire Aiguë (SDRA),
    sur la biomécanique et la physiologie pulmonaire. Cette maladie, qui touche près de 30 % des patients en réanimation, 
	présente un taux de mortalité très élevé (30 à 40 %). Les anesthésistes-réanimateurs qui prennent en charge ces patients 
    éprouvent des difficultés à leur administrer la ventilation mécanique la mieux adaptée pour maximiser leurs chances de survie. 
    En effet, cette pathologie est très « patient-spécifique » et reste largement méconnue. Afin de mieux comprendre cette pathologie 
    et ses effets sur la biomécanique pulmonaire, des tests ont été effectués pour étudier le comportement mécanique du tissu atteint 
    par le SDRA. Parallèlement, des échantillons de tissu ont été prélevés pour effectuer une analyse histologique visant à établir
    un lien entre le comportement mécanique et la microstructure (cellules et matrice extracellulaire). Ces tests ont notamment été 
    réalisés par Ombeline Juteau, doctorante au LBA.</p> 
	<p>Il est crucial d'étudier ces caractéristiques car le SDRA entraîne une dégradation profonde de l'architecture pulmonaire : 
	d'une part, la réponse inflammatoire provoque un remodelage de la matrice extracellulaire, notamment via une accumulation ou 
    une désorganisation du collagène ; d'autre part, l'œdème et le collapsus alvéolaire modifient drastiquement la distribution 
    de l'air dans le parenchyme. Comprendre ces changements structurels est essentiel pour la recherche, car cela permet 
    d'identifier comment l'altération des composants biologiques modifie les propriétés mécaniques globales du poumon.</p>
	<p>Un exemple d'image histologique ainsi qu'une illustration de la segmentation
	automatique des cellules sont fournis dans <a href="examples.php" class="secondary-button">la page d'exemples</a>.</p>
</section>

<hr>

<section class="features">
    <h3>⚙️ Fonctionnalités Clés</h3>
    <div class="feature-list">
        <div>Segmentation automatique du collagène, du tissu et des zones d’air.</div>
        <div>Exclusion automatique du fond externe pour des mesures précises.</div>
        <div>Quantification des surfaces relatives de chaque composante.</div>
        <div>Traitement en série de plusieurs images à pleine résolution.</div>
    </div>
</section>

<section id="visual-demo" class="content-section centered">
    <h3>Voir les résultats de la segmentation</h3>
    <p>Visualisez la puissance de l'analyse automatique sur des échantillons histologiques.</p>
    <a href="examples.php" class="secondary-button">Galerie d'Exemples</a>
</section>

<?php
    // Inclure le Footer
    include 'includes/footer.php';
?>