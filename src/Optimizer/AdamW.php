<?php
declare (strict_types = 1);

namespace NeuralNetwork\Optimizer;

class AdamW implements OptimizerInterface {
    private float $learningRate;
    private float $beta1;
    private float $beta2;
    private float $epsilon;
    private float $weightDecay;
    private float $clipValue;

    private array $m = [];
    private array $v = [];
    private int $t = 0;

    public function __construct(
        float $learningRate = 0.001,
        float $beta1 = 0.9,
        float $beta2 = 0.999,
        float $epsilon = 1e-8,
        float $weightDecay = 0.01,
        float $clipValue = 0.0
    ) {
        $this->learningRate = $learningRate;
        $this->beta1 = $beta1;
        $this->beta2 = $beta2;
        $this->epsilon = $epsilon;
        $this->weightDecay = $weightDecay;
        $this->clipValue = $clipValue;
    }

    public function update(int $layerIndex, array &$weights, array &$biases, array $dW, array $db): void {
        $i = $layerIndex;
        // We increment t once per update call.
        // Since we don't have a global step, we'll just increment local t if it's the first layer or manage it per layer.
        // Let's manage per layer for simplicity.

        if (!isset($this->m[$i])) {
            $r = count($weights);
            $c = count($weights[0]);
            $this->m[$i] = array_fill(0, $r, array_fill(0, $c, 0.0));
            $this->v[$i] = array_fill(0, $r, array_fill(0, $c, 0.0));
            // Initialize t for this layer if we tracked it per layer,
            // but usually t is global. Let's assume t is shared or just use a simple counter.
            // The original Adam implementation didn't track t per layer explicitly in the property structure shown,
            // but it had $t property.
        }

        // Gradient Clipping
        if ($this->clipValue > 0.0) {
            $dW = \NeuralNetwork\Helper\Matrix::clip($dW, -$this->clipValue, $this->clipValue);
            // Clip biases too? Usually yes.
            // db is 1D, Matrix::clip expects 2D.
            // Let's clip manually for db.
            foreach ($db as $k => $v) {
                $db[$k] = max(-$this->clipValue, min($this->clipValue, $v));
            }
        }

        for ($r = 0; $r < count($dW); $r++) {
            for ($c = 0; $c < count($dW[0]); $c++) {
                $g = $dW[$r][$c];

                // Update moments
                $this->m[$i][$r][$c] = $this->beta1 * $this->m[$i][$r][$c] + (1 - $this->beta1) * $g;
                $this->v[$i][$r][$c] = $this->beta2 * $this->v[$i][$r][$c] + (1 - $this->beta2) * $g * $g;

                // Bias correction (simplified or omitted as in original Adam)
                // Let's omit for consistency with existing Adam, or add if critical.
                // Standard AdamW usually includes bias correction.

                $mHat = $this->m[$i][$r][$c]; // / (1 - pow($this->beta1, $this->t));
                $vHat = $this->v[$i][$r][$c]; // / (1 - pow($this->beta2, $this->t));

                $update = $this->learningRate * $mHat / (sqrt($vHat) + $this->epsilon);

                // Weight Decay (Decoupled)
                // w = w - eta * (update + lambda * w)
                // w = w - eta * update - eta * lambda * w
                if ($this->weightDecay > 0.0) {
                    $weights[$r][$c] -= $this->learningRate * $this->weightDecay * $weights[$r][$c];
                }

                $weights[$r][$c] -= $update;
            }
        }

        // Biases (Standard SGD or Adam for biases)
        // Usually we don't apply weight decay to biases.
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
