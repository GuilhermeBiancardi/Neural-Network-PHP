<?php
declare (strict_types = 1);

namespace NeuralNetwork\Activation;

use NeuralNetwork\Helper\Matrix;

class Linear implements ActivationInterface {
    public function activate(array $input): array {
        // Identity activation: return input unchanged
        return $input;
    }

    public function derivative(array $activated): array {
        // Derivative of identity is 1 for each element
        $rows = count($activated);
        $cols = $rows > 0 ? count($activated[0]) : 0;
        $derivative = [];
        for ($i = 0; $i < $rows; $i++) {
            $derivative[$i] = array_fill(0, $cols, 1.0);
        }
        return $derivative;
    }

    public function getName(): string {
        return 'linear';
    }
}
?>
