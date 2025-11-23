<?php
declare (strict_types = 1);

require_once __DIR__ . '/../NeuralNetwork.class.php';

use NeuralNetwork\Layer\Dense;

// Generate sine wave data
$inputs = [];
$targets = [];
for ($i = 0; $i < 100; $i++) {
    $x = $i / 10.0;
    $inputs[] = [$x];
    $targets[] = [sin($x)];
}

// Build network: 1 input -> hidden layers -> output 1 neuron
$layers = [
    new Dense(1, 32, 'tanh'),
    new Dense(32, 32, 'tanh'),
    new Dense(32, 1, 'linear'),
];

$nn = new \NeuralNetwork($layers);
$nn->configure([
    'learning_rate' => 0.01,
    'optimizer' => 'adam',
    'loss' => 'mse',
]);

$nn->train($inputs, $targets, 5000, 0, true);

// Test predictions
for ($i = 0; $i < 5; $i++) {
    $x = ($i + 10) / 10.0;
    $pred = $nn->predict([$x]);
    echo "x=$x => predicted=" . $pred[0] . ", actual=" . sin($x) . "\n";
}
?>
