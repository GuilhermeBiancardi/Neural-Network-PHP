<?php
declare (strict_types = 1);

namespace NeuralNetwork\Activation;

class HardSwish implements ActivationInterface {
    public function activate(array $input): array {
        $output = [];
        foreach ($input as $row) {
            $newRow = [];
            foreach ($row as $val) {
                // x * HardSigmoid(x) = x * max(0, min(1, (x + 3) / 6))
                // if x <= -3: 0
                // if x >= 3: x
                // else: x * (x + 3) / 6
                if ($val <= -3.0) {
                    $newRow[] = 0.0;
                } elseif ($val >= 3.0) {
                    $newRow[] = $val;
                } else {
                    $newRow[] = $val * ($val + 3.0) / 6.0;
                }
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
                // if x < -3: 0
                // if x > 3: 1
                // else: (2x + 3) / 6 = x/3 + 0.5
                if ($val < -3.0) {
                    $newRow[] = 0.0;
                } elseif ($val > 3.0) {
                    $newRow[] = 1.0;
                } else {
                    $newRow[] = $val / 3.0 + 0.5;
                }
            }
            $output[] = $newRow;
        }
        return $output;
    }

    public function getName(): string {
        return 'hardswish';
    }
}
