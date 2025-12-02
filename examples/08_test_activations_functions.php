<?php
require_once __DIR__ . '/../vendor/autoload.php';

use NeuralNetwork\Activation\Gelu;
use NeuralNetwork\Activation\HardSigmoid;
use NeuralNetwork\Activation\HardSwish;
use NeuralNetwork\Activation\HardTanh;
use NeuralNetwork\Activation\LeakyRelu;
use NeuralNetwork\Activation\Relu;
use NeuralNetwork\Activation\Sigmoid;
use NeuralNetwork\Activation\Tanh;

$activations = [
    new Gelu(),
    new HardSigmoid(),
    new HardSwish(),
    new HardTanh(),
    new Sigmoid(),
    new Tanh(),
    new Relu(),
    new LeakyRelu(),
];

$input = [[-3.0, -1.0, 0.0, 1.0, 3.0]];

foreach ($activations as $activation) {
    echo "Testing " . $activation->getName() . "...\n";
    $output = $activation->activate($input);
    $derivative = $activation->derivative($input); // Passing Z

    echo "Input: " . implode(", ", $input[0]) . "\n";
    echo "Output: " . implode(", ", $output[0]) . "\n";
    echo "Derivative: " . implode(", ", $derivative[0]) . "\n";
    echo "--------------------------------------------------\n";
}
