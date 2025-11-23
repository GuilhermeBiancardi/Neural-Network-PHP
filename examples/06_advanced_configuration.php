<?php
declare (strict_types = 1);

require_once __DIR__ . '/../NeuralNetwork.class.php';

use NeuralNetwork\Layer\BatchNormalization;
use NeuralNetwork\Layer\Dense;
use NeuralNetwork\Layer\Dropout;

// Simple regression dataset (y = 2x + 1)
$inputs = [];
$targets = [];
for ($i = 0; $i < 200; $i++) {
    $x = $i / 10.0;
    $inputs[] = [$x];
    $targets[] = [2 * $x + 1];
}

// Build network with advanced layers
$layers = [
    new Dense(1, 64, 'relu'),
    new BatchNormalization(64),
    new Dropout(0.1), // Dropout rate of 10%
    new Dense(64, 1, 'linear'), // assuming 'linear' is identity
];

$nn = new \NeuralNetwork($layers);
$nn->configure([
    'learning_rate' => 0.01,
    'optimizer' => 'adam',
    'loss' => 'mse',
    'batch_size' => 32,
]);

$nn->train($inputs, $targets, 1000, 0, true);

// Test predictions
for ($i = 0; $i < 5; $i++) {
    $x = ($i + 210) / 10.0;
    $pred = $nn->predict([$x]);
    echo "x=$x => predicted=" . $pred[0] . ", actual=" . (2 * $x + 1) . "\n";
}
?>
