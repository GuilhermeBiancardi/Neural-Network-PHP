<?php
declare (strict_types = 1);

namespace NeuralNetwork\Activation;

interface ActivationInterface {
    public function activate(array $input): array;
    public function derivative(array $input): array;
    public function getName(): string;
}
