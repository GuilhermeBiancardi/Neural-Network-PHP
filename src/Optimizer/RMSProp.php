<?php
declare (strict_types = 1);

namespace NeuralNetwork\Optimizer;

class RMSProp implements OptimizerInterface {
    private float $learningRate;
    private float $beta;
    private float $epsilon;
    private array $cache = [];

    public function __construct(float $learningRate = 0.001, float $beta = 0.9, float $epsilon = 1e-8) {
        $this->learningRate = $learningRate;
        $this->beta = $beta;
        $this->epsilon = $epsilon;
    }

    public function update(int $layerIndex, array &$weights, array &$biases, array $dW, array $db): void {
        $i = $layerIndex;

        if (!isset($this->cache[$i])) {
            $r = count($weights);
            $c = count($weights[0]);
            $this->cache[$i] = array_fill(0, $r, array_fill(0, $c, 0.0));
        }

        for ($r = 0; $r < count($dW); $r++) {
            for ($c = 0; $c < count($dW[0]); $c++) {
                $g = $dW[$r][$c];

                $this->cache[$i][$r][$c] = $this->beta * $this->cache[$i][$r][$c] + (1 - $this->beta) * $g * $g;

                $update = $this->learningRate * $g / (sqrt($this->cache[$i][$r][$c]) + $this->epsilon);
                $weights[$r][$c] -= $update;
            }
        }

        for ($j = 0; $j < count($db); $j++) {
            $biases[$j] -= $this->learningRate * $db[$j];
        }
    }

    public function reset(): void {
        $this->cache = [];
    }
}
