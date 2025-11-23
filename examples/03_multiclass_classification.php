<?php
declare (strict_types = 1);

require_once __DIR__ . '/../NeuralNetwork.class.php';

use NeuralNetwork\Layer\Dense;

// Simple dummy Iris-like dataset (features: petal length, petal width)
$inputs = [
    [1.4, 0.2], // Setosa
    [1.5, 0.2], // Setosa
    [4.7, 1.4], // Versicolor
    [4.5, 1.5], // Versicolor
    [5.9, 2.3], // Virginica
    [6.0, 2.5], // Virginica
];
$targets = [
    [1, 0, 0], // Setosa
    [1, 0, 0],
    [0, 1, 0], // Versicolor
    [0, 1, 0],
    [0, 0, 1], // Virginica
    [0, 0, 1],
];

// Build network: 2 inputs -> hidden 5 neurons -> output 3 neurons
$layers = [
    new Dense(2, 5, 'relu'),
    new Dense(5, 3, 'softmax'),
];

$nn = new \NeuralNetwork($layers);
$nn->configure([
    'learning_rate' => 0.05,
    'optimizer' => 'adam',
    'loss' => 'crossentropy',
]);

$nn->train($inputs, $targets, 3000, 0, true);

// Test predictions
foreach ($inputs as $idx => $input) {
    $output = $nn->predict($input);
    $predClass = array_search(max($output), $output);
    echo "Sample $idx predicted class: $predClass, probabilities: " . implode(', ', $output) . "\n";
}
?>
