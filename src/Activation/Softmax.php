<?php
declare (strict_types = 1);

namespace NeuralNetwork\Activation;

class Softmax implements ActivationInterface {
    public function activate(array $input): array {
        $result = [];
        // Input is (Features x BatchSize)
        // We apply Softmax column-wise (per sample)
        $cols = count($input[0]);
        $rows = count($input);

        for ($j = 0; $j < $cols; $j++) {
            $col = [];
            for ($i = 0; $i < $rows; $i++) {
                $col[] = $input[$i][$j];
            }

            $max = max($col); // Stability
            $sum = 0.0;
            $exp = [];
            foreach ($col as $val) {
                $e = exp($val - $max);
                $exp[] = $e;
                $sum += $e;
            }

            for ($i = 0; $i < $rows; $i++) {
                $result[$i][$j] = $exp[$i] / $sum;
            }
        }
        return $result;
    }

    public function derivative(array $input): array {
        // Softmax derivative is complex (Jacobian).
        // However, when combined with CrossEntropy, the delta is simply (Output - Target).
        // If this is called, it returns 1.0 as a placeholder, assuming the Loss function handles the delta calculation correctly.
        // This is a common optimization/simplification in simple implementations.
        $result = [];
        for ($i = 0; $i < count($input); $i++) {
            for ($j = 0; $j < count($input[0]); $j++) {
                $result[$i][$j] = 1.0;
            }
        }
        return $result;
    }

    public function getName(): string {
        return 'softmax';
    }
}
