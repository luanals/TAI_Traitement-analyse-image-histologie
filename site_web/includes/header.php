<?php
// On démarre la session si elle n'est pas déjà démarrée
if (session_status() === PHP_SESSION_NONE) {
    session_start();
}
?>
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Analyse automatisée d’images histologiques pulmonaires pour le SDRA. Quantification du collagène, tissu et air.">
    <meta name="keywords" content="SDRA, ARDS, Histologie, Polytech, Biomédical, Python, Analyse d'images">
    <meta name="author" content="Alcide Demeusy et Luana Lopes Santiago">
    
    <title>TAI-SDRA – Analyse Histologique Pulmonaire</title>
    
    <link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
    <header>
        <div class="container">
            <h1>TAI-SDRA</h1>
            <nav>
				<ul>
    				<li><a href="index.php">Accueil</a></li>
    				<li><a href="about.php">À Propos</a></li>
    				<li><a href="usage.php">Utilisation</a></li>
    				<li><a href="examples.php">Exemples</a></li>
    				<li><a href="contact.php">Contact</a></li>

    				<?php if (isset($_SESSION['est_connecte']) && $_SESSION['est_connecte'] === true): ?>
        				<li class="locked-link"><a href="processus.php">🔓 Processus</a></li>
    				<?php else: ?>
        				<li class="locked-link"><a href="processus.php">🔒 Processus</a></li>
    				<?php endif; ?>
				</ul>
            </nav>
        </div>
    </header>
    <main class="container">