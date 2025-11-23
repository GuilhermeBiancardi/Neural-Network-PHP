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

$layers = [
    new Dense(2, 8, 'relu'),
    new Dense(8, 1, 'sigmoid'),
];

$nn = new \NeuralNetwork($layers);
$nn->configure([
    'learning_rate' => 0.1,
    'optimizer' => 'adam',
    'loss' => 'mse',
]);

$nn->train($inputs, $targets, 3000, 0, true);

// Save model
$modelPath = __DIR__ . '/xor_model.dat';
$nn->save($modelPath);

echo "Model saved to $modelPath\n";

// Load model
$loaded = NeuralNetwork::load($modelPath);

echo "Loaded model predictions:\n";
foreach ($inputs as $input) {
    $out = $loaded->predict($input);
    echo "Input: [" . implode(', ', $input) . "] => Predicted: " . round($out[0]) . " (raw: " . $out[0] . ")\n";
}
?>
