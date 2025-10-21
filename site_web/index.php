<?php
    // Optionnel: Définir un titre spécifique pour cette page
    $page_title = "Aperçu du Projet | TAI-SDRA";
    
    // Inclure le Header
    include 'includes/header.php';
?>

<section class="hero">
    <h2>Analyse automatisée d’images histologiques pulmonaires – SDRA</h2>
    <p class="lead">Un outil développé pour mieux comprendre l'impact du <b>Syndrome de Détresse Respiratoire Aiguë (SDRA)</b> sur la biomécanique pulmonaire à partir d’images colorées au <b>Trichrome de Masson</b>.</p>
</section>

<section class="objectives">
    <h3>🎯 Objectifs Principaux</h3>
    <ul>
        <li>Mise en place d'un <b>processus automatisé d’analyse</b> pour identifier et quantifier les composantes (collagène, tissu, air).</li>
        <li>Fournir une <b>évaluation quantitative fiable</b> des proportions relatives.</li>
        <li>Exporter les résultats sous forme exploitable (`.csv`) pour des analyses statistiques.</li>
    </ul>
</section>

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