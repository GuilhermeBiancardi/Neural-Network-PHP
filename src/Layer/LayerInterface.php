<?php
declare (strict_types = 1);

namespace NeuralNetwork\Layer;

use NeuralNetwork\Optimizer\OptimizerInterface;

interface LayerInterface {
    /**
     * Forward pass.
     * @param array $input Input data (batch_size x input_dim)
     * @param bool $training Whether we are training or predicting
     * @return array Output data
     */
    public function forward(array $input, bool $training = false): array;

    /**
     * Backward pass.
     * @param array $outputGradient Gradient of loss w.r.t output
     * @return array Gradient of loss w.r.t input
     */
    public function backward(array $outputGradient): array;

    /**
     * Update parameters using the optimizer.
     * @param OptimizerInterface $optimizer
     * @param int $layerIndex
     */
    public function updateParams(OptimizerInterface $optimizer, int $layerIndex): void;

    /**
     * Get layer parameters (weights, biases, etc.) for serialization.
     * @return array
     */
    public function getParams(): array;

    /**
     * Load layer parameters.
     * @param array $params
     */
    public function setParams(array $params): void;

    /**
     * Get the output shape of the layer given an input shape.
     * @param array $inputShape
     * @return array
     */
    public function getOutputShape(array $inputShape): array;
}
