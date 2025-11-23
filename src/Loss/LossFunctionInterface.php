<?php
declare (strict_types = 1);

namespace NeuralNetwork\Loss;

interface LossFunctionInterface {
    public function calculate(array $output, array $target): float;
    public function calculateGradient(array $output, array $target): array;
}
