<?php
declare (strict_types = 1);

namespace NeuralNetwork\Activation;

class HardTanh implements ActivationInterface {
    private float $minVal;
    private float $maxVal;

    public function __construct(float $minVal = -1.0, float $maxVal = 1.0) {
        $this->minVal = $minVal;
        $this->maxVal = $maxVal;
    }

    public function activate(array $input): array {
        $output = [];
        foreach ($input as $row) {
            $newRow = [];
            foreach ($row as $val) {
                // max(min, min(max, x))
                $newRow[] = max($this->minVal, min($this->maxVal, $val));
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
                // 1 if min <= x <= max, else 0
                if ($val >= $this->minVal && $val <= $this->maxVal) {
                    $newRow[] = 1.0;
                } else {
                    $newRow[] = 0.0;
                }
            }
            $output[] = $newRow;
        }
        return $output;
    }

    public function getName(): string {
        return 'hardtanh';
    }
}
