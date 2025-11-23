<?php
declare (strict_types = 1);

namespace NeuralNetwork\Loss;

use NeuralNetwork\Helper\Matrix;

class CrossEntropy implements LossFunctionInterface {
    public function calculate(array $output, array $target): float {
        // -sum(target * log(output))
        // Add epsilon to avoid log(0)
        $epsilon = 1e-15;
        $sum = 0.0;
        $count = count($output[0]); // Batch size

        for ($i = 0; $i < count($output); $i++) {
            for ($j = 0; $j < count($output[0]); $j++) {
                $y = $target[$i][$j];
                $yHat = $output[$i][$j];
                $yHat = max($epsilon, min(1 - $epsilon, $yHat)); // Clip
                $sum += -($y * log($yHat));
            }
        }
        return $sum / $count;
    }

    public function calculateGradient(array $output, array $target): array {
        // For Softmax + CrossEntropy, the gradient w.r.t Z (input to activation) is (Output - Target).
        // However, the backprop loop usually does: delta = error * activation_derivative.
        // If activation is Softmax, its derivative() returns 1.0 (placeholder).
        // So if we return (Output - Target) here, the multiplication by 1.0 gives the correct delta.
        return Matrix::subtract($output, $target);
    }
}
