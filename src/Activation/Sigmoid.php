<?php
declare (strict_types = 1);

namespace NeuralNetwork\Activation;

class Sigmoid implements ActivationInterface {
    public function activate(array $input): array {
        $result = [];
        for ($i = 0; $i < count($input); $i++) {
            for ($j = 0; $j < count($input[0]); $j++) {
                $result[$i][$j] = 1.0 / (1.0 + exp(-$input[$i][$j]));
            }
        }
        return $result;
    }

    public function derivative(array $input): array {
        // Input here is expected to be the OUTPUT of the activation (sigmoid(z))
        // f'(x) = f(x) * (1 - f(x))
        $result = [];
        for ($i = 0; $i < count($input); $i++) {
            for ($j = 0; $j < count($input[0]); $j++) {
                $val = $input[$i][$j];
                $result[$i][$j] = $val * (1.0 - $val);
            }
        }
        return $result;
    }

    public function getName(): string {
        return 'sigmoid';
    }
}
