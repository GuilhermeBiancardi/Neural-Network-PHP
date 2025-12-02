<?php
declare (strict_types = 1);

namespace NeuralNetwork\Activation;

class HardSigmoid implements ActivationInterface {
    public function activate(array $input): array {
        $output = [];
        foreach ($input as $row) {
            $newRow = [];
            foreach ($row as $val) {
                // max(0, min(1, (x + 3) / 6))
                $newRow[] = max(0.0, min(1.0, ($val + 3.0) / 6.0));
            }
            $output[] = $newRow;
        }
        return $output;
    }

    public function derivative(array $input): array {
        $output = [];
        foreach ($input as $row) {
            $newRow = [];
            foreach ($row as $val) {
                // 1/6 if -3 <= x <= 3, else 0
                if ($val >= -3.0 && $val <= 3.0) {
                    $newRow[] = 1.0 / 6.0;
                } else {
                    $newRow[] = 0.0;
                }
            }
            $output[] = $newRow;
        }
        return $output;
    }

    public function getName(): string {
        return 'hardsigmoid';
    }
}
