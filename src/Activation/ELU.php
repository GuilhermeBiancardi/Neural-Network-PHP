<?php
declare (strict_types = 1);

namespace NeuralNetwork\Activation;

class ELU implements ActivationInterface {
    private float $alpha;

    public function __construct(float $alpha = 1.0) {
        $this->alpha = $alpha;
    }

    public function activate(array $input): array {
        $output = [];
        foreach ($input as $row) {
            $newRow = [];
            foreach ($row as $val) {
                $newRow[] = $val > 0 ? $val : $this->alpha * (exp($val) - 1);
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
                // f'(x) = 1 if x > 0
                // f'(x) = f(x) + alpha if x <= 0
                // We have f(x) in $val.
                $newRow[] = $val > 0 ? 1.0 : $val + $this->alpha;
            }
            $output[] = $newRow;
        }
        return $output;
    }

    public function getName(): string {
        return 'elu';
    }
}
