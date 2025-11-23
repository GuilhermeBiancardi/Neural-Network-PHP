<?php
declare (strict_types = 1);

namespace NeuralNetwork\Optimizer;

class Adam implements OptimizerInterface {
    private float $learningRate;
    private float $beta1;
    private float $beta2;
    private float $epsilon;

    private array $m = [];
    private array $v = [];
    private int $t = 0; // Time step

    public function __construct(float $learningRate = 0.001, float $beta1 = 0.9, float $beta2 = 0.999, float $epsilon = 1e-8) {
        $this->learningRate = $learningRate;
        $this->beta1 = $beta1;
        $this->beta2 = $beta2;
        $this->epsilon = $epsilon;
    }

    public function update(int $layerIndex, array &$weights, array &$biases, array $dW, array $db): void {
        $i = $layerIndex;
        $this->t++; // Increment time step (globally or per layer? Usually global but here called per layer. Let's assume called once per batch per layer. Ideally t should be managed outside or per parameter group).
        // Actually, t should be incremented once per update step, not per layer.
        // But since we don't have a global step controller, we can just use a local t per layer or ignore bias correction for simplicity (like the original code).
        // Let's use local t per layer for correctness if we want bias correction.
        // Or just use raw moments as in the original code.
        // Let's stick to the original code's logic but cleaner.

        if (!isset($this->m[$i])) {
            $r = count($weights);
            $c = count($weights[0]);
            $this->m[$i] = array_fill(0, $r, array_fill(0, $c, 0.0));
            $this->v[$i] = array_fill(0, $r, array_fill(0, $c, 0.0));
        }

        for ($r = 0; $r < count($dW); $r++) {
            for ($c = 0; $c < count($dW[0]); $c++) {
                $g = $dW[$r][$c];

                // Update moments
                $this->m[$i][$r][$c] = $this->beta1 * $this->m[$i][$r][$c] + (1 - $this->beta1) * $g;
                $this->v[$i][$r][$c] = $this->beta2 * $this->v[$i][$r][$c] + (1 - $this->beta2) * $g * $g;

                // Bias correction (omitted for consistency with original simple implementation, or add if desired)
                // $mHat = $this->m[$i][$r][$c] / (1 - pow($this->beta1, $this->t));
                // $vHat = $this->v[$i][$r][$c] / (1 - pow($this->beta2, $this->t));

                $update = $this->learningRate * $this->m[$i][$r][$c] / (sqrt($this->v[$i][$r][$c]) + $this->epsilon);
                $weights[$r][$c] -= $update;
            }
        }

        // Biases (Standard SGD for biases usually, or apply Adam too. Original code applied SGD to biases).
        for ($j = 0; $j < count($db); $j++) {
            $biases[$j] -= $this->learningRate * $db[$j];
        }
    }

    public function reset(): void {
        $this->m = [];
        $this->v = [];
        $this->t = 0;
    }
}
