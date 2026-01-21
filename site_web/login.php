<?php
// On démarre la session au tout début
session_start();

// Si l'utilisateur est déjà connecté, on le redirige directement vers la page protégée
if (isset($_SESSION['est_connecte']) && $_SESSION['est_connecte'] === true) {
    header("Location: processus.php");
    exit;
}

$erreur = "";

// Traitement du formulaire
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $mot_de_passe_saisi = $_POST['password'];
    
    // --- CONFIGURATION DU MOT DE PASSE ICI ---
    $mot_de_passe_correct = "polytech"; 

    if ($mot_de_passe_saisi === $mot_de_passe_correct) {
        // Mot de passe correct : on enregistre l'état dans la session
        $_SESSION['est_connecte'] = true;
        header("Location: processus.php");
        exit;
    } else {
        $erreur = "Mot de passe incorrect.";
    }
}
?>

<?php include 'includes/header.php'; ?>

<section class="login-container">
    <h2>🔒 Accès Restreint</h2>
    <p>Cette page détaille notre méthodologie interne. Veuillez vous identifier.</p>

    <div class="login-box">
        <?php if (!empty($erreur)) echo "<p class='error-msg'>$erreur</p>"; ?>
        
        <form method="POST" action="login.php">
            <label for="password">Mot de passe :</label>
            <input type="password" id="password" name="password" required>
            <button type="submit">Se Connecter</button>
        </form>
    </div>
</section>

<?php include 'includes/footer.php'; ?>