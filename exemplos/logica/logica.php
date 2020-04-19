<?php

/**
 * Nesse exemplo iremos testar a capacidade da rede em detectar padrões
 * onde criei uma lógica simples:
 * 
 * A rede receberá 5 entradas, onde o valor dessas entradas são valores
 * 0.1 (que consideraremos sendo igual a 1), se informar-mos as e entradas
 * como 0.1 então a rede deverá nos informar o número 5 pois se semar-mos
 * todos os valores de entrada (5 valores 0.1 que equivalem a 1) daria 5.
 * 
 * Se informarmos 4 valores a rede deve retornar o valor 4 e assim por
 * diante!
 */

include_once "../../class/NeuralNetwork.class.php";

$nn = new NeuralNetwork();

$rede = Array (
    "structure" => Array(5, 5, 5),
    "error" => 0.01,
    "memory_limit" => "4096M"
);

$nn->setConfiguration($rede);

// Valores de entrada
$values = Array(
    Array(0.1, 0.1, 0.1, 0.1, 0.1),
    Array(0.1, 0.1, 0.1, 0.1, 0),
    Array(0.1, 0.1, 0.1, 0, 0),
    Array(0.1, 0.1, 0, 0, 0),
    Array(0.1, 0, 0, 0, 0),
);

/**
 * Respostas esperadas onde 1 é o resultado correto
 * e 0 é o resultado que não está correto.
 */
$response = Array(
    //    1, 2, 3, 4, 5
    Array(0, 0, 0, 0, 1),
    Array(0, 0, 0, 1, 0),
    Array(0, 0, 1, 0, 0),
    Array(0, 1, 0, 0, 0),
    Array(1, 0, 0, 0, 0),
);

// Treinamos a rede
foreach($values as $key => $value) {
    $nn->setValues($values[$key]);
    $nn->setResponse($response[$key]);
    $nn->train();
}

// Verificamos se a rede aprendeu a ler o número 5
echo "Pedindo para ela ler o número 5" . PHP_EOL;
$nn->setValues($values[0]);
foreach($nn->answerBinary() as $key => $value) {
    $numero = round($value);
    if($numero >= 1) {
        echo "Resposta da Rede: " . ($key +1) . PHP_EOL;
    }
}

// Verificamos se a rede aprendeu a ler o número 4
echo "Pedindo para ela ler o número 4" . PHP_EOL;
$nn->setValues($values[1]);
foreach($nn->answerBinary() as $key => $value) {
    $numero = round($value);
    if($numero >= 1) {
        echo "Resposta da Rede: " . ($key +1) . PHP_EOL;
    }
}

// Verificamos se a rede aprendeu a ler o número 3
echo "Pedindo para ela ler o número 3" . PHP_EOL;
$nn->setValues($values[2]);
foreach($nn->answerBinary() as $key => $value) {
    $numero = round($value);
    if($numero >= 1) {
        echo "Resposta da Rede: " . ($key +1) . PHP_EOL;
    }
}

// Verificamos se a rede aprendeu a ler o número 2
echo "Pedindo para ela ler o número 2" . PHP_EOL;
$nn->setValues($values[3]);
foreach($nn->answerBinary() as $key => $value) {
    $numero = round($value);
    if($numero >= 1) {
        echo "Resposta da Rede: " . ($key +1) . PHP_EOL;
    }
}

// Verificamos se a rede aprendeu a ler o número 1
echo "Pedindo para ela ler o número 1" . PHP_EOL;
$nn->setValues($values[4]);
foreach($nn->answerBinary() as $key => $value) {
    $numero = round($value);
    if($numero >= 1) {
        echo "Resposta da Rede: " . ($key +1) . PHP_EOL;
    }
}

/**
 * Aqui faremos uma lógica diferente para ver se nossa rede
 * sacou o padrão que criamos, iremos mudar a posição de alguns
 * valores e substituiremos um dos valores 0.1 por 0.2 e zeraremos
 * outro, segundo nossa lógica 0.2 equivaleria ao número 2 exemplo:
 * 
 * 0 + 0.1 + 0.1 + 0.1 + 0.2 = 0 + 1 + 1 + 1 + 2 = 5.
 * 
 * Será que a rede vai convergir certo?
 */

// Verificamos se a rede aprendeu a ler o novo número 5
echo "Pedindo para ela ler o número 5 NOVO" . PHP_EOL;
$nn->setValues(Array(0, 0.1, 0.1, 0.2, 0.1));
foreach($nn->answerBinary() as $key => $value) {
    $numero = round($value);
    if($numero >= 1) {
        echo "Resposta da Rede: " . ($key +1) . PHP_EOL;
    }
}

// Verificamos se a rede aprendeu a ler o novo número 4
echo "Pedindo para ela ler o número 4 NOVO" . PHP_EOL;
$nn->setValues(Array(0.1, 0.1, 0.2, 0, 0));
foreach($nn->answerBinary() as $key => $value) {
    $numero = round($value);
    if($numero >= 1) {
        echo "Resposta da Rede: " . ($key +1) . PHP_EOL;
    }
}