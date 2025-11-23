<?php
declare (strict_types = 1);

namespace NeuralNetwork\Activation;

class Relu implements ActivationInterface {
    public function activate(array $input): array {
        $result = [];
        for ($i = 0; $i < count($input); $i++) {
            for ($j = 0; $j < count($input[0]); $j++) {
                $result[$i][$j] = max(0.0, $input[$i][$j]);
            }
        }
        return $result;
    }

    public function derivative(array $input): array {
        // Input here is expected to be the OUTPUT of the activation?
        // Actually, for ReLU derivative we need Z (input to activation) or check if A > 0.
        // If A = ReLU(Z), then if A > 0, Z > 0. So checking A is sufficient.
        $result = [];
        for ($i = 0; $i < count($input); $i++) {
            for ($j = 0; $j < count($input[0]); $j++) {
                $result[$i][$j] = $input[$i][$j] > 0 ? 1.0 : 0.0;
            }
        }
        return $result;
    }

    public function getName(): string {
        return 'relu';
    }
}
