<?php include 'includes/header.php'; ?>

<section class="visuals">
    <h2>Exemples de Segmentation Visuelle</h2>
    <p>L'algorithme de segmentation identifie et colore les différentes composantes pulmonaires (Coll. : bleu, Tissu : rouge, Air : noir) pour une validation visuelle rapide.</p>

    <div class="example-image-container">
        <div class="image-pair">
            <img src="assets/img/original_histology.jpg" alt="Image histologique originale (Trichrome de Masson)">
            <p>Image Originale</p>
        </div>
        <div class="image-pair">
            <img src="assets/img/segmented_output.png" alt="Image segmentée et colorée">
            <p>Résultat de la Segmentation</p>
        </div>
        <p><img src="assets/img/test.png" alt="image test"/></p>
    </div>
    
</section>

<hr>

<section class="quantitative">
    <h2>Résultats Quantitatifs</h2>
    <p>Le tableau ci-dessous présente un exemple d'évaluation quantitative fournie par l'outil, exportée dans un fichier `.csv` :</p>

    <table>
        <thead>
            <tr>
                <th>Fichier Image</th>
                <th>Coll. (%)</th>
                <th>Tissu (%)</th>
                <th>Air (%)</th>
                <th>Surface Totale (px�)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>sample_001.tif</td>
                <td>12.5</td>
                <td>25.3</td>
                <td>62.2</td>
                <td>3,456,000</td>
            </tr>
            <tr>
                <td>sample_002.png</td>
                <td>9.8</td>
                <td>28.1</td>
                <td>62.1</td>
                <td>4,100,000</td>
            </tr>
            </tbody>
    </table>
</section>

<?php include 'includes/footer.php'; ?>