<?php
    $page_title = "Résultats & Démonstration";
    include 'includes/header.php';
?>

<section class="hero-results">
    <h2>📊 Résultats Visuels & Quantitatifs</h2>
    <p class="lead">
        Validation visuelle de l'algorithme <strong>TAI</strong> sur des échantillons haute définition.
    </p>
</section>

<section class="reading-guide">
    <h3>🎨 Guide de Lecture (Trichrome de Masson)</h3>
    <p>Que ce soit sur l'image originale ou segmentée, les couleurs ont une signification biologique :</p>
    
    <div class="legend-container">
        <div class="legend-item">
            <span class="color-dot blue"></span>
            <div>
                <strong>Bleu = Collagène</strong>
                <span class="sub-text">Marqueur de la Fibrose (Pathologie)</span>
            </div>
        </div>
        <div class="legend-item">
            <span class="color-dot red"></span>
            <div>
                <strong>Rouge/Violet = Tissu, à l’exclusion du collagène</strong>
                <span class="sub-text">Muscle, Cytoplasme, Cellules</span>
            </div>
        </div>
        <div class="legend-item">
            <span class="color-dot white"></span>
            <div>
                <strong>Blanc = Air</strong>
                <span class="sub-text">Alvéoles pulmonaires ou Vide</span>
            </div>
        </div>
    </div>
</section>

<hr>

<section class="zoom-showcase">
    <h3>🔍 Inspection HD : Comparaison de Pathologies</h3>
    <p>
        L'algorithme doit être capable de discriminer les tissus sains des tissus fibrosés. 
        Voici une comparaison entre des zone avec plus et moins de collagène.
    </p>

    <div class="case-study">
        <h4 class="case-title">🔴 Cas A : Fibrose Plus Sévère (Beaucoup de Collagène)</h4>
        <div class="zoom-container">
            <div class="zoom-box">
                <img src="assets/img/zoom_high_orig.png" alt="Cas A Original">
                <span class="img-label">Original (Trichrome)</span>
            </div>
            <div class="zoom-box">
                <img src="assets/img/zoom_high_seg.jpg" alt="Cas A Segmenté">
                <span class="img-label">Segmentation</span>
            </div>
        </div>
        <div class="result-bar">
            <span>Score Fibrose détecté : <strong>54.79%</strong></span>
            <div class="progress-track"><div class="progress-fill high" style="width: 45%;"></div></div>
        </div>
    </div>

    <hr class="separator-dashed">

    <div class="case-study">
        <h4 class="case-title">🟢 Cas B : Fibrose Plus Faible</h4>
        <div class="zoom-container">
            <div class="zoom-box">
                <img src="assets/img/zoom_low_orig.png" alt="Cas B Original">
                <span class="img-label">Original (Trichrome)</span>
            </div>
            <div class="zoom-box">
                <img src="assets/img/zoom_low_seg.jpg" alt="Cas B Segmenté">
                <span class="img-label">Segmentation</span>
            </div>
        </div>
        <div class="result-bar">
            <span>Score Fibrose détecté : <strong>36.38%</strong></span>
            <div class="progress-track"><div class="progress-fill low" style="width: 12%;"></div></div>
        </div>
    </div>

</section>

<hr>

<section class="smart-edges">
    <h3>✨ Amélioration des contours ("Smart Edges")</h3>
    <div class="comparison-container">
        <div class="text-content">
            <p>
                Une difficulté majeure des images WSI est la gestion des bords du poumon.
                Les anciennes versions créaient un effet "escalier". La <strong>version finale</strong> utilise une interpolation adaptative pour lisser ces contours.
            </p>
            <ul class="specs-list">
                <li><strong>Avant :</strong> Contours pixelisés, risque de faux positifs.</li>
                <li><strong>Après (version finale) :</strong> Lissage naturel qui suit la membrane.</li>
            </ul>
        </div>
        <div class="visual-content">
            <img src="assets/img/smart_edges_comparaison.png" alt="Comparaison Escalier vs Lisse (Page 12)">
            <p class="caption">Comparaison : Masque binaire classique vs Interpolation de la version finale</p>
        </div>
    </div>
</section>

<hr>

<section class="global-stats">
    <h3>📈 Analyse Quantitative (19 Images)</h3>
    <p>Distribution des résultats sur le jeu de données complet.</p>

    <div class="stats-dashboard">
        <div class="stat-card">
            <span class="stat-value">5.2 Md</span>
            <span class="stat-label">Pixels Analysés</span>
        </div>
        <div class="stat-card">
            <span class="stat-value">49.2 %</span>
            <span class="stat-label">Fibrose Max</span>
        </div>
    </div>

    <div class="boxplot-container">
        <img src="assets/img/boxplot_global.png" alt="Statistiques Globales (Page 15)">
    </div>
</section>

<?php include 'includes/footer.php'; ?>