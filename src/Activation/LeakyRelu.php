<?php
declare (strict_types = 1);

namespace NeuralNetwork\Activation;

class LeakyRelu implements ActivationInterface {
    private float $alpha;

    public function __construct(float $alpha = 0.01) {
        $this->alpha = $alpha;
    }

    public function activate(array $input): array {
        $output = [];
        foreach ($input as $row) {
            $newRow = [];
            foreach ($row as $val) {
                $newRow[] = $val > 0 ? $val : $this->alpha * $val;
            }
            $output[] = $newRow;
        }
        return $output;
    }

    public function derivative(array $activatedInput): array {
        $output = [];
        foreach ($activatedInput as $row) {
            $newRow = [];
            foreach ($row as $val) {
                $newRow[] = $val > 0 ? 1.0 : $this->alpha;
            }
            $output[] = $newRow;
        }
        return $output;
    }

    public function getName(): string {
        return 'leakyrelu';
    }
}
