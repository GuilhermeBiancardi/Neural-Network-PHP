<?php

declare (strict_types = 1);

namespace NeuralNetwork\Optimizer;

class SGD implements OptimizerInterface {
    private float $learningRate;
    private float $momentum;
    private float $weightDecay;
    private float $clipValue;
    private array $prevDeltaWeights = [];

    public function __construct(float $learningRate = 0.1, float $momentum = 0.0, float $weightDecay = 0.0, float $clipValue = 0.0) {
        $this->learningRate = $learningRate;
        $this->momentum = $momentum;
        $this->weightDecay = $weightDecay;
        $this->clipValue = $clipValue;
    }

    public function update(int $layerIndex, array &$weights, array &$biases, array $dW, array $db): void {
        $i = $layerIndex;

        // Initialize momentum state if needed
        if (!isset($this->prevDeltaWeights[$i])) {
            $r = count($weights);
            $c = count($weights[0]);
            $this->prevDeltaWeights[$i] = array_fill(0, $r, array_fill(0, $c, 0.0));
        }

        // Gradient Clipping
        if ($this->clipValue > 0.0) {
            $dW = \NeuralNetwork\Helper\Matrix::clip($dW, -$this->clipValue, $this->clipValue);
            foreach ($db as $k => $v) {
                $db[$k] = max(-$this->clipValue, min($this->clipValue, $v));
            }
        }

        // Calculate update term
        // Update = lr * (dW + decay * W)
        // If momentum > 0:
        // v = momentum * v + update
        // W = W - v

        // 1. Apply Weight Decay to Gradient
        if ($this->weightDecay > 0.0) {
            $decayTerm = \NeuralNetwork\Helper\Matrix::scalarMultiply($weights, $this->weightDecay);
            $dW = \NeuralNetwork\Helper\Matrix::add($dW, $decayTerm);
        }

        // 2. Calculate base update
        $update = \NeuralNetwork\Helper\Matrix::scalarMultiply($dW, $this->learningRate);

        // 3. Apply Momentum
        if ($this->momentum > 0) {
            $momentumTerm = \NeuralNetwork\Helper\Matrix::scalarMultiply($this->prevDeltaWeights[$i], $this->momentum);
            $update = \NeuralNetwork\Helper\Matrix::add($momentumTerm, $update);
            $this->prevDeltaWeights[$i] = $update;
        }

        // 4. Update Weights
        $weights = \NeuralNetwork\Helper\Matrix::subtract($weights, $update);

        // Biases
        for ($j = 0; $j < count($db); $j++) {
            $biases[$j] -= $this->learningRate * $db[$j];
        }
    }

    public function reset(): void {
        $this->prevDeltaWeights = [];
    }
}