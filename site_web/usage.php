<?php include 'includes/header.php'; ?>

<section class="installation">
    <h2>🖥️ Installation du Projet</h2>
    <p>Pour utiliser l'outil d'analyse, suivez les étapes ci-dessous. Le script principal est en Python.</p>
    
    <ol>
        <li>
            <strong>1. Cloner le dépôt :</strong>
            <pre><code>git clone https://github.com/ton-utilisateur/TAI-SDRA.git
cd TAI-SDRA</code></pre>
        </li>
        <li>
            <strong>2. Créer un environnement virtuel (recommandé) :</strong>
            <pre><code>python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows</code></pre>
        </li>
        <li>
            <strong>3. Installer les dépendances Python :</strong>
            <pre><code>pip install -r requirements.txt</code></pre>
        </li>
    </ol>
</section>

<hr>

<section class="running">
    <h2>🚀 Exécution du Script</h2>
    
    <ol>
        <li>Placer les images histologiques (`.tif`, `.jpg`, `.png`, etc.) dans un dossier dédié (ex: `/data/input`).</li>
        <li>Exécuter le script principal en spécifiant les chemins d'entrée et de sortie :
            <pre><code>python analyse_pulmonaire.py --input /chemin/vers/images --output resultats/resultats.csv</code></pre>
        </li>
        <li>Les résultats (pourcentages de collagène, tissu, air) seront enregistrés dans le fichier `.csv` spécifié.</li>
    </ol>
</section>

<hr>

<section class="structure">
    <h2>📁 Structure du Projet</h2>
    <pre>
.
├── analyse_pulmonaire.py       # Script principal (traitement et analyse)
├── exemples/                   # Images de test
├── resultats/                  # Résultats exportés (.csv)
├── site_web/                   # Scripts du site web (ce que vous construisez)
├── requirements.txt            # Dépendances Python
└── README.md
    </pre>
</section>

<?php include 'includes/footer.php'; ?>