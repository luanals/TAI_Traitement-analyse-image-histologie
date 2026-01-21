<?php
session_start();

// VÉRIFICATION DE SÉCURITÉ
if (!isset($_SESSION['est_connecte']) || $_SESSION['est_connecte'] !== true) {
    header("Location: login.php");
    exit;
}
?>

<?php 
	$page_title = "Notre processus | TAI-SDRA";
	include 'includes/header.php'; 
?>

<section class="workflow">
	<div class="page-navigation">
    <h2>📑 Sommaire</h2>
    <ul>
        <li><a href="#phase1">1. Le Pipeline Actuel</a></li>
        <li><a href="#phase2">2. Historique (ImageJ/Matlab)</a></li>
        <li><a href="#phase3">3. Optimisation RAM</a></li>
        <li><a href="#phase4">4. Solution Finale (v51)</a></li>
    </ul>
	</div>
	<hr>
    <h2 style="border-bottom: 2px solid #eee; padding-bottom: 15px; margin-bottom: 30px;">
        🛠️ Journal de Bord & Processus Technique
    </h2>

    <p class="intro-text">
        Cette page documente notre démarche. Elle sert de rapport technique et de guide pour les futurs étudiants reprenant le projet TAI-SDRA.
    </p>

    <h3 id="phase1" class="section-title">📂 1. Le Pipeline Actuel (Python)</h3>
    <div class="timeline">
        <div class="step">
            <div class="step-number">1</div>
            <div class="step-content">
                <h3>Acquisition & Chargement</h3>
                <p>Nous travaillons sur des images <strong>.tiff</strong> (contrairement au format .ndpi de l'année précédente). En raison de la taille massive des images (WSI ~70k x 50k px), nous utilisons une stratégie de <em>Downscale Preview</em> pour charger l'image en mémoire sans faire crasher la RAM (Facteur recommandé : 2 ou 3).</p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">2</div>
            <div class="step-content">
                <h3>Segmentation Colorimétrique</h3>
                <p>L'algorithme sépare les structures basées sur la coloration Trichrome de Masson. Nous avons calibré les seuils pour isoler le <strong>collagène (Bleu)</strong>, <strong>le tissu sain (Rouge/Magenta)</strong> et le <strong>fond/air (Blanc/Vide)</strong>.</p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">3</div>
            <div class="step-content">
                <h3>Optimisation & Quantification</h3>
                <p>Pour accélérer le temps de traitement (passer de >40 min à ~2 min par image), nous appliquons un <em>Downscale Factor</em> lors du comptage de pixels. Nos tests (voir ci-dessous) montrent qu'un facteur de 10 offre le meilleur compromis vitesse/précision (&lt; 1% d'écart).</p>
            </div>
        </div>
        
        <div class="step">
            <div class="step-number">4</div>
            <div class="step-content">
                <h3>Export & Visualisation</h3>
                <p>Génération automatique d'un fichier <code>.csv</code> compatible Excel pour l'analyse statistique et création d'une image de contrôle avec les zones segmentées pour validation visuelle.</p>
            </div>
        </div>
    </div>

    <hr style="margin: 50px 0;">

    <h3 id="phase2" class="section-title">📝 2. Retours d'Expérience (Ce qu'on a essayé)</h3>
    
    <div class="technical-log">
        <h4>🔍 Phase 1 : Exploration des outils (ImageJ & Matlab)</h4>
        <p>Avant de développer notre propre solution Python, nous avons évalué les outils existants.</p>
        <ul>
            <li><strong>ImageJ / Fiji :</strong> Tentative avec le plugin <em>Color Deconvolution</em>.
                <br><em>Problème :</em> Les fichiers WSI (Whole Slide Images) sont trop lourds. Le format YCbCr des .tiff posait problème pour la déconvolution standard (nécessite RGB). Les images s'ouvraient souvent totalement noires ou nécessitaient un crop manuel fastidieux (Coordonnées testées : X=17920, Y=29440).</li>
            <li><strong>Matlab :</strong> Utilisation du script existant.
                <br><em>Résultat :</em> Fonctionnel pour le prototypage (Proportion collagène ~3.59%), mais difficilement automatisable en "batch" (série) sans licence Matlab coûteuse pour l'utilisateur final. L'objectif étant l'Open Source + à cause des solutions déjà existantes, nous avons pivoté vers Python.</li>
        </ul>

        <h4 id="phase3" >⚠️ Phase 2 : Le Mur de la RAM (Optimisation)</h4>
        <p>Lors du passage à Python, nous avons rencontré des erreurs de mémoire (<code>MemoryError</code>) et des temps de calculs prohibitifs (> 40 minutes par image).</p>
        <p><strong>Constat :</strong> L'utilisation du GPU n'a pas résolu le problème car le goulot d'étranglement était la RAM (Mémoire vive) et non la puissance de calcul brute. Il faut charger 60 Go de données dans 16 Go de RAM.</p>
        
        <h4>☑️ Phase 3 : La Solution (Downscaling)</h4>
        <p>Nous avons mis en place une double stratégie de réduction d'échelle. Voici le tableau comparatif de nos tests pour guider les futurs utilisateurs :</p>
        
        <table class="data-table">
            <thead>
                <tr>
                    <th>Stratégie</th>
                    <th>Config Testée</th>
                    <th>RAM Estimée</th>
                    <th>Temps / Image</th>
                    <th>Précision</th>
                    <th>Verdict</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Brut</strong></td>
                    <td>Preview 1 / Factor 1</td>
                    <td>~60 Go</td>
                    <td>Crash 💥</td>
                    <td>100%</td>
                    <td>❌ Impossible</td>
                </tr>
                <tr>
                    <td><strong>Mode "Sûr"</strong></td>
                    <td>Preview 3 / Factor 10</td>
                    <td>~6 Go</td>
                    <td>~2 min</td>
                    <td>±1.5%</td>
                    <td>✅ Stable (Rec. pour vieux PC)</td>
                </tr>
                <tr>
                    <td><strong>Mode "Équilibré"</strong></td>
                    <td>Preview 2 / Factor 10</td>
                    <td>~12 Go</td>
                    <td>~2-3 min</td>
                    <td>±1.0%</td>
                    <td>⭐ <strong>Recommandé (PC 16Go RAM)</strong></td>
                </tr>
                <tr>
                    <td><strong>Haute Précision</strong></td>
                    <td>Preview 2 / Factor 7</td>
                    <td>~12 Go</td>
                    <td>~5 min</td>
                    <td>±0.5%</td>
                    <td>⚠️ Long pour peu de gain</td>
                </tr>
            </tbody>
        </table>
        <p class="note"><em>Note : Le "Downscale Factor" de 10 signifie que nous analysons 1 pixel sur 100 (10x10), ce qui accélère drastiquement le calcul tout en gardant une erreur statistique négligeable sur des images de cette taille.</em></p>
                                                                           
        <h4 id="phase4" >✅ Phase 4 : La Version Finale (V51 "Smart Edges")</h4>
        <p>Ce qu'on a gardé du code précédent Matlab :<p>
        <ul><li>L'idée de sélection des plages de couleurs de chaque image par histogramme.</li></ul>
        <p>Notre dernière itération, la <strong>v51</strong>, résout deux problèmes majeurs identifiés lors des tests précédents :</p>        
        <ul>
            <li>
                <strong>Problème des bordures en escalier :</strong> Le masque "Radar" basse définition créait des contours pixelisés lors du passage en haute définition.
                <br><em>Solution ("Smart Edges") :</em> Utilisation d'une <strong>interpolation bilinéaire</strong> (<code>order=1</code>) lors du redimensionnement du masque, couplée à un seuil souple (0.15). Cela lisse les contours du poumon et évite de couper artificiellement les tissus en bordure.
            </li>
            <li>
                <strong>Problème du Collagène pâle :</strong> Certaines coupes présentent un collagène très clair (Cyan) que l'algorithme ignorait.
                <br><em>Solution ("Cyan Force") :</em> Implémentation d'une calibration dynamique (V36) qui force la détection du spectre Cyan (Teinte > 0.48) tout en bloquant strictement le spectre Violet du muscle (Teinte &lt; 0.85).
            </li>
        </ul>
                                                                           
        <h3 class="section-title">La Calibration "Cyan Force"</h3>
        <p>
            Pour garantir la robustesse de l'analyse, nous ne pouvons pas utiliser des seuils de couleurs fixes (ex: "Bleu > 150"). La chimie des colorants varie d'une lame à l'autre.
        </p>

        <div class="calibration-grid">
            <div class="calib-img">
                <img src="assets/img/calibration_v36.png" alt="Graphique de Calibration V36">
            </div>
            <div class="calib-desc">
                <h4>Notre solution (Algorithme v36) :</h4>
                <p>Avant chaque analyse, le script scanne l'image et génère un histogramme des teintes (voir ci-contre).</p>
                <ul>
                    <li><strong>1. Détection du Pic :</strong> L'algo trouve où se situe le bleu principal.</li>
                    <li><strong>2. Cyan Force (Min 0.48) :</strong> On force l'inclusion du cyan très clair (souvent du collagène fin).</li>
                    <li><strong>3. Mur Violet (Max 0.85) :</strong> On interdit strictement le dépassement vers le violet pour ne pas confondre avec le muscle.</li>
                </ul>
            </div>
        </div>
                                                                           
    </div>
</section>

<?php include 'includes/footer.php'; ?>