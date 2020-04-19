<?php

/**
 * Neste arquivo iremos treinar a rede com base
 * em algumas imagens de letras, iremos testar
 * se a rede consegue indentificar imagens!
 * 
 * Iremos informar 81 entradas que equivale a cada
 * pixel da imagem (cada imagem é 9x9) e 7 saidas
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

include_once "../../class/ImageData.class.php";
include_once "../../class/NeuralNetwork.class.php";

$img = new ImageData();
$nn = new NeuralNetwork();

$file = "../../arquivos/image.data";

$rede = Array (
    "structure" => Array(81, 15, 7),
    "error" => 0.01,
    "error_learn" => 0.01,
    "memory_limit" => "4096M"
);

$nn->setConfiguration($rede);

// Tipos de fontes a serem treinadas
$images = Array(
    "../../images/calibri_a.png",
    "../../images/calibri_b.png",
    "../../images/calibri_c.png",
    "../../images/calibri_d.png",
    "../../images/segoe_a.png",
    "../../images/segoe_b.png",
    "../../images/segoe_c.png",
    "../../images/segoe_d.png",
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

// Treino a rede
foreach($values as $key => $value) {
    $nn->setValues($values[$key]);
    $nn->setResponse($response[$key]);    
    $nn->train();
}

/**
 * Varificamos se a rede aprendeu cada letra corretamente,
 * a resposta esperada é a sequencia de A,B,C,D,A,B,C,D.
 */ 
foreach($values as $value) {
    $nn->setValues($value);
    $char = "";
    foreach($nn->answerBinary() as $v) {
        $char .= round($v);
    }
    echo "Resposta da Rede: " . chr(bindec($char)) . PHP_EOL;
}

/**
 * Como o processo de treino demora irei exportar o resultado
 * do treino e salva-lo em um arquivo, para não precisar treinar
 * a rede novamente quando for testa-la.
 */

$data = fopen($file, "w");

fwrite($data, $nn->exportData());

fclose($data);

?>