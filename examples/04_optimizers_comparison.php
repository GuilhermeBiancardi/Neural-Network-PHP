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

$optimizers = ['sgd', 'adam', 'rmsprop'];
foreach ($optimizers as $opt) {
    $layers = [
        new Dense(2, 2, 'relu'),
        new Dense(2, 1, 'sigmoid'),
    ];
    $nn = new \NeuralNetwork($layers);
    $nn->configure([
        'learning_rate' => 0.1,
        'optimizer' => $opt,
        'loss' => 'mse',
    ]);
    echo "Training with optimizer: $opt\n";
    $nn->train($inputs, $targets, 3000, 0, true);
    // Evaluate loss after training
    $output = [];
    foreach ($inputs as $inp) {
        $output[] = $nn->predict($inp);
    }
    // Simple mean squared error calculation
    $mse = 0.0;
    foreach ($output as $i => $pred) {
        $mse += ($pred[0] - $targets[$i][0]) ** 2;
    }
    $mse /= count($targets);
    echo "Final MSE for $opt: $mse\n\n";
}
?>
