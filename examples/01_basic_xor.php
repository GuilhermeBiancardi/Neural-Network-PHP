<?php
declare (strict_types = 1);

require_once __DIR__ . '/../NeuralNetwork.class.php';

use NeuralNetwork\Layer\Dense;

// XOR dataset
$inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
];
$targets = [
    [0],
    [1],
    [1],
    [0],
];

// Build network: 2 inputs -> hidden 8 neurons -> output 1 neuron
$layers = [
    new Dense(2, 8, 'relu'),
    new Dense(8, 1, 'sigmoid'),
];

$nn = new \NeuralNetwork($layers);
$nn->configure([
    'learning_rate' => 0.01,
    'optimizer' => 'adam',
    'loss' => 'mse',
]);

$nn->train($inputs, $targets, 10000, 0, true);

// Test predictions
foreach ($inputs as $input) {
    $output = $nn->predict($input);
    echo "Input: [" . implode(', ', $input) . "] => Predicted: " . round($output[0]) . " (raw: " . $output[0] . "\n";
}
?>
