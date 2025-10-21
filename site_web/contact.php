<?php
/**
 * Logique PHP pour la gestion du formulaire de contact.
 */
$message_status = "";

if (isset($_POST['submit'])) {
    // 1. Définir l'adresse de réception
    $to = "luana.LOPES-SANTIAGO@etu.univ-amu.fr";
    
    // 2. Récupérer et sécuriser les données
    $subject = "Nouveau contact TAI-SDRA: " . htmlspecialchars($_POST['subject']);
    $name = htmlspecialchars($_POST['name']);
    $email = filter_var($_POST['email'], FILTER_VALIDATE_EMAIL);
    $message = htmlspecialchars($_POST['message']);

    if (!$email) {
        $message_status = "? L'adresse email fournie n'est pas valide.";
    } elseif (empty($name) || empty($subject) || empty($message)) {
        $message_status = "? Veuillez remplir tous les champs du formulaire.";
    } else {
        // 3. Construire le contenu du mail
        $email_content = "Nom: $name\n";
        $email_content .= "Email: $email\n\n";
        $email_content .= "Message:\n$message\n";
    
        // 4. Définir les en-têtes
        $email_headers = "From: " . $email . "\r\n" .
                         "Reply-To: " . $email . "\r\n" .
                         "X-Mailer: PHP/" . phpversion();
    
        // 5. Utiliser la fonction mail() (nécessite un serveur configuré)
        if (mail($to, $subject, $email_content, $email_headers)) {
            $message_status = "? Votre message a été envoyé avec succès ! Nous vous répondrons bientôt.";
            // Optionnel : Vider les champs après succès si vous ne voulez pas les réafficher
            unset($_POST['name'], $_POST['email'], $_POST['subject'], $_POST['message']);
        } else {
            $message_status = "? Erreur critique lors de l'envoi. Veuillez vérifier la configuration de mail() sur le serveur.";
        }
    }
}
?>

<?php include 'includes/header.php'; ?>

<section class="contact-form">
    <h2 style="text-align:center;">Nous Contacter</h2>
    <p>Pour toute question technique, collaboration ou demande d'information sur le projet SDRA, veuillez utiliser le formulaire ci-dessous.</p>
    
    <?php 
    // Affichage des messages de statut (succès ou erreur)
    if (!empty($message_status)) { 
        echo "<p class='status-message'>" . $message_status . "</p>"; 
    } 
    ?>
    
    <form method="POST" action="contact.php" class="centered-form">
    	<div class="form-group">
        <label for="name">Nom / Organisation :</label>
        <input type="text" id="name" name="name" required value="<?php echo isset($_POST['name']) ? htmlspecialchars($_POST['name']) : ''; ?>">
		</div>
        
        <div class="form-group">
        <label for="email">Votre Email :</label>
        <input type="email" id="email" name="email" required value="<?php echo isset($_POST['email']) ? htmlspecialchars($_POST['email']) : ''; ?>">
		</div>
        
        <div class="form-group">
        <label for="subject">Sujet :</label>
        <input type="text" id="subject" name="subject" required value="<?php echo isset($_POST['subject']) ? htmlspecialchars($_POST['subject']) : ''; ?>">
		</div>
        
        <div class="form-group">
        <label for="message">Message :</label>
        <textarea id="message" name="message" rows="6" required><?php echo isset($_POST['message']) ? htmlspecialchars($_POST['message']) : ''; ?></textarea>
        </div>

        <button type="submit" name="submit">Envoyer le Message</button>
    </form>
</section>

<?php include 'includes/footer.php'; ?>