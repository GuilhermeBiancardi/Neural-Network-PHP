<?php
declare (strict_types = 1);

namespace NeuralNetwork\Activation;

class Gelu implements ActivationInterface {

    public function activate(array $input): array {
        $output = [];
        foreach ($input as $row) {
            $newRow = [];
            foreach ($row as $val) {
                // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                $newRow[] = 0.5 * $val * (1 + tanh(sqrt(2 / M_PI) * ($val + 0.044715 * pow($val, 3))));
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
                // f'(x) = 0.5 * (1 + tanh(y)) + 0.5 * x * sech^2(y) * dy/dx
                // where y = sqrt(2/pi) * (x + 0.044715 * x^3)
                // dy/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)

                $x = $val;
                $c = sqrt(2 / M_PI);
                $y = $c * ($x + 0.044715 * pow($x, 3));
                $tanhY = tanh($y);

                $dy_dx = $c * (1 + 3 * 0.044715 * pow($x, 2));
                $sech2 = 1.0 - $tanhY * $tanhY; // sech^2(y) = 1 - tanh^2(y)

                $derivative = 0.5 * (1 + $tanhY) + 0.5 * $x * $sech2 * $dy_dx;
                $newRow[] = $derivative;
            }
            $output[] = $newRow;
        }
        return $output;
    }

    public function getName(): string {
        return 'gelu';
    }
}
