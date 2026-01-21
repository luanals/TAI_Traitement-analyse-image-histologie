</main>
    <footer>
        <div class="container">
			<img src="assets/img/amu.png" alt="Logo Aix-Marseille Université">
			<img src="assets/img/amu-Polytech.png" alt="Logo Polytech Marseille">
			<img src="assets/img/LBA.png" alt="Logo LBA">
            <p>&copy; <?php echo date("Y"); ?> TAI-SDRA Project. Réalisé par Alcide Demeusy & Luana Lopes Santiago.</p>
            <p>Polytech Aix-Marseille Université</p>
            
            <div class="w3c-badge">
    			<?php
        			$protocol = (!empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off' || $_SERVER['SERVER_PORT'] == 443) ? "https://" : "http://";

    				$current_url = $protocol . $_SERVER['HTTP_HOST'] . $_SERVER['REQUEST_URI'];

    			    $encoded_url = urlencode($current_url);
    			?>

    			<a href="https://validator.w3.org/check?uri=<?php echo $encoded_url; ?>" target="_blank">
        			<img style="border:0;width:88px;height:31px"
            			src="https://www.w3.org/Icons/valid-html401-blue" 
            			alt="Valid HTML!">
    			</a>

    			<br> <a href="https://jigsaw.w3.org/css-validator/validator?uri=https://luana-lopes-santiago-etu.pedaweb.univ-amu.fr/extranet/TAI-SDRA/assets/css/style.css" target="_blank">
        			<img style="border:0;width:88px;height:31px"
            			src="https://jigsaw.w3.org/css-validator/images/vcss-blue"
            			alt="Valid CSS!">
    			</a>
			</div>         
        </div>
    </footer>
</body>
</html>