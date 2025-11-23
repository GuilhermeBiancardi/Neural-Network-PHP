<?php
declare (strict_types = 1);

namespace NeuralNetwork\Loss;

use NeuralNetwork\Helper\Matrix;

class MSE implements LossFunctionInterface {
    public function calculate(array $output, array $target): float {
        $diff = Matrix::subtract($target, $output);
        $sum = 0.0;
        $count = 0;
        foreach ($diff as $row) {
            foreach ($row as $val) {
                $sum += $val * $val;
                $count++;
            }
        }
        return $sum / $count; // Mean Squared Error
    }

    public function calculateGradient(array $output, array $target): array {
        // dC/dA = -(Target - Output) = Output - Target
        // But we usually return the error term that will be multiplied by activation derivative.
        // For MSE, Error = (Output - Target).
        return Matrix::subtract($output, $target);
    }
}
