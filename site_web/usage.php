<?php include 'includes/header.php'; ?>

<section class="terminal">
	<h2>En utilisant le terminal :</h2>
    <h2>🖥️ Installation du Projet</h2>
    <p>Pour utiliser l'outil d'analyse, suivez les étapes ci-dessous. Le script principal est en Python.</p>
    
    <ol>
        <li>
            <strong> Cloner le dépôt :</strong>
            <pre><code>git clone https://github.com/ton-utilisateur/TAI-SDRA.git
cd TAI-SDRA</code></pre>
        </li>
        <li>
            <strong> Créer un environnement virtuel (recommandé) :</strong>
            <pre><code>python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows</code></pre>
        </li>
        <li>
            <strong> Installer les dépendances Python :</strong>
            <pre><code>pip install -r requirements.txt</code></pre>
        </li>
    </ol>
    <h2>🚀 Exécution du Script</h2>
    
    <ol>
        <li>Placer les images histologiques (`.tif`, `.jpg`, `.png`, etc.) dans un dossier dédié (ex: `/data/input`).</li>
        <li>Exécuter le script principal (choisissez le script avec scikit-learn ou OpenCV) en spécifiant les chemins d'entrée et de sortie :
            <pre><code>python analyse_pulmonaire-SI.py --input /chemin/vers/images --output resultats/resultats.csv</code></pre>
            <pre><code>python analyse_pulmonaire-OCV.py --input /chemin/vers/images --output resultats/resultats.csv</code></pre>
        </li>
        <li>Les résultats (pourcentages de collagène, tissu, air) seront enregistrés dans le fichier `.csv` spécifié.</li>
    </ol>        
</section>

<hr>

<section class="lowcode">
	    <h2>Pas utilisant le terminal :</h2>
        <h2>🖥️ Installation du Projet</h2>
   <ol>
        	<li><strong>Assurez-vous d'avoir téléchargé les fichier requirements.txt et le(s) script(s) principal analyse_pulmonaire.py que vous voulez utiliser</strong></li>
            <li><strong> Installer les dépendances Python :</strong>
            <pre><code>pip install -r requirements.txt</code></pre></li>
   </ol>
        <h2>🚀 Exécution du Script</h2>
    
    <ol>
        <li>Placer les images histologiques (`.tif`, `.jpg`, `.png`, etc.) dans un dossier dédié (ex: `/data/input`).</li>
        <li>Exécuter le script principal (choisissez le script avec scikit-learn ou OpenCV) dans votre plataforme de préférence :</li>
    </ol>
    <ul>
        <p>Le script vous guidera à travers une interface simple :</p>
		<li>Chemin de l'image : Entrez le chemin complet de votre fichier TIFF.</li>
		<li>Dossier de sortie : Entrez le dossier où vous souhaitez enregistrer les résultats (par défaut, le dossier de l'image).</li>
	</ul>
</section>

<hr>

<section class="structure">
    <h2>📁 Structure du Projet</h2>
    <pre>
.
├── analyse_pulmonaire-OCV.py   # Script principal (traitement et analyse)
├── analyse_pulmonaire-SI.py    # Script principal (traitement et analyse)        
├── exemples/                   # Images de test
├── resultats/                  # Résultats exportés (.csv)
├── site_web/                   # Scripts du site web
├── requirements.txt            # Dépendances Python
└── README.md
    </pre>
</section>

<?php include 'includes/footer.php'; ?>
