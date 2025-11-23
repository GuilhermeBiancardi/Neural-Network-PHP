<?php
declare (strict_types = 1);

namespace NeuralNetwork\Activation;

class Tanh implements ActivationInterface {
    public function activate(array $input): array {
        $result = [];
        for ($i = 0; $i < count($input); $i++) {
            for ($j = 0; $j < count($input[0]); $j++) {
                $result[$i][$j] = tanh($input[$i][$j]);
            }
        }
        return $result;
    }

    public function derivative(array $input): array {
        // f'(x) = 1 - f(x)^2
        $result = [];
        for ($i = 0; $i < count($input); $i++) {
            for ($j = 0; $j < count($input[0]); $j++) {
                $val = $input[$i][$j];
                $result[$i][$j] = 1.0 - ($val * $val);
            }
        }
        return $result;
    }

    public function getName(): string {
        return 'tanh';
    }
}
