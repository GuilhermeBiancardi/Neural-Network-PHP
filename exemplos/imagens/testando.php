<?php

include_once "../../class/ImageData.class.php";
include_once "../../class/NeuralNetwork.class.php";

$img = new ImageData();
$nn = new NeuralNetwork();

// Pego os dados de treino do arquivo

$file = "../../arquivos/image.data";

$data = fopen($file, "r");

$linha = "";

while(!feof($data)) {
    $linha .= fgets($data, 1024);
}

fclose($data);

// Importo os dados de treino capturados
$nn->importData($linha);

// Imagens não treinadas a serem testadas
$images_new = Array(
    "A" => "../../images/lucida_a.png",
    "B" => "../../images/lucida_b.png",
    "C" => "../../images/lucida_c.png",
    "D" => "../../images/lucida_d.png",
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
    $nn->setValues($value);
    $char = "";
    foreach($nn->answerBinary() as $v) {
        $char .= round($v);
    }
    echo "Saída da Rede: " . chr(bindec($char)) . PHP_EOL;
}