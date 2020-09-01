<?php

include_once "../../../Class/ImageData.class.php";
include_once "../../../Neural/Structure.php";

/**
 * Neste arquivo iremos treinar a rede com base
 * em algumas imagens de letras, iremos testar
 * se a rede consegue identificar imagens!
 * 
 * Iremos informar 81 entradas que equivale a cada
 * pixel da imagem (cada imagem é 9x9) e 7 saídas
 * equivalentes a cada caractere binário correspondente
 * a letra contida na imagem ex:
 * 
 * A = 1000001
 * B = 1000010
 * C = 1000011
 * D = 1000100
 * 
 * Então a saída esperada será o conjunto de binários
 * da letra.
 */

$img = new ImageData();
$nn = new BackPropagation();

$images = Array(

    // Fonte Calibri
    "../../../images/calibri_a.png",
    "../../../images/calibri_b.png",
    "../../../images/calibri_c.png",
    "../../../images/calibri_d.png",

    // Fonte Segoe
    "../../../images/segoe_a.png",
    "../../../images/segoe_b.png",
    "../../../images/segoe_c.png",
    "../../../images/segoe_d.png",

);

$values = Array();

/**
 * Popula o array de entrada com os valores de cada imagem
 * com o auxilio da classe ImagemDada que nos dara as
 * informações de cada pixel das imagens.
 */
foreach($images as $value) {
    $img->setImage($value);
    $values[] = $img->getImageInfo();
}

// Respostas esperadas com base em cada imagem.
$response = Array(

    // Resultados para Calibri
    Array(1, 0, 0, 0, 0, 0, 1), // A
    Array(1, 0, 0, 0, 0, 1, 0), // B
    Array(1, 0, 0, 0, 0, 1, 1), // C
    Array(1, 0, 0, 0, 1, 0, 0), // D

    // Resultados para Segoe
    Array(1, 0, 0, 0, 0, 0, 1), // A
    Array(1, 0, 0, 0, 0, 1, 0), // B
    Array(1, 0, 0, 0, 0, 1, 1), // C
    Array(1, 0, 0, 0, 1, 0, 0), // D

);

$nn->prepareStructure([81,30,7], [new Sigmoid(), new Sigmoid()], new Bias());

// Treino a rede
for($loop = 0; $loop < 1000; $loop++) {
    for($i = 0; $i < count($values); $i++) {
        $nn->setInputs($values[$i]);
        $nn->setExpectedResponse($response[$i]);    
        $nn->train();
    }
}

// Imagens não treinadas a serem testadas
$images_new = Array(
    "A" => "../../../images/lucida_a.png",
    "B" => "../../../images/lucida_b.png",
    "C" => "../../../images/lucida_c.png",
    "D" => "../../../images/lucida_d.png",
);

$values_new = Array();

/**
 * Popula o array de entrada com os valores de cada imagem
 * com o auxilio da classe ImagemDada que nos dara as
 * informações de cada pixel das imagens.
 */
foreach($images_new as $key => $value) {
    $img->setImage($value);
    $values_new[$key] = $img->getImageInfo();
}

// Resultado obtido:
foreach($values_new as $key => $value) {
    echo "Correspondente a letra: " . $key . PHP_EOL;
    
    $nn->setInputs($value);
    $output = $nn->getResponse();

    $bin = "";
    foreach($output as $saida) {
        $bin .= round($saida[0]);
    }
    echo chr(bindec($bin)) . PHP_EOL;
}

?>