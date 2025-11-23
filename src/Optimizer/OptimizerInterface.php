<?php
declare (strict_types = 1);

namespace NeuralNetwork\Optimizer;

interface OptimizerInterface {
    public function update(int $layerIndex, array &$weights, array &$biases, array $dW, array $db): void;
    public function reset(): void;
}
