<?php
declare (strict_types = 1);

namespace NeuralNetwork\Layer;

use NeuralNetwork\Optimizer\OptimizerInterface;

class Flatten implements LayerInterface {
    private array $inputShape = [];

    public function forward(array $input, bool $training = false): array {
        // Input: [batch_size, d1, d2, ...] or [batch_size, d1]
        // We want to flatten everything after batch_size.

        $batchSize = count($input);
        if ($batchSize === 0) {
            return [];
        }

        // Store shape for backward
        // Assuming all samples have same shape
        // We can't easily store full shape if it's jagged, but NN inputs are usually tensors.
        // Let's assume input[0] structure represents the rest.
        // Actually, we just need to know how to reshape back.
        // But PHP arrays are tricky.
        // Let's store the original input for backward if needed, or just the dimensions.
        // Storing full input might be memory intensive.
        // Let's store the dimensions of a single sample.

        $this->inputShape = $this->getShape($input[0]);

        $output = [];
        foreach ($input as $sample) {
            $output[] = $this->flatten($sample);
        }

        return $output;
    }

    private function flatten(array $array): array {
        $result = [];
        array_walk_recursive($array, function ($a) use (&$result) {
            $result[] = $a;
        });
        return $result;
    }

    private function getShape(array $array): array {
        $shape = [];
        $current = $array;
        while (is_array($current)) {
            $shape[] = count($current);
            if (empty($current)) {
                break;
            }

            $current = $current[0];
        }
        return $shape;
    }

    public function backward(array $outputGradient): array {
        // outputGradient: [batch_size, flattened_size]
        // We need to reshape it back to inputShape.

        $inputGradient = [];
        foreach ($outputGradient as $grad) {
            $inputGradient[] = $this->reshape($grad, $this->inputShape);
        }
        return $inputGradient;
    }

    private function reshape(array $flat, array $shape): array {
        // Recursively reconstruct the array
        if (empty($shape)) {
            return $flat;
        }
        // Should be a scalar if fully recursed, but here we might have logic error if shape is empty.

        // Actually, if shape is [10, 10], we take 10 chunks of size 10.
        // If shape is [3, 10, 10], we take 3 chunks of size (10*10).

        $dim = array_shift($shape);
        if (empty($shape)) {
            return $flat; // Base case? No, if shape was [10], we return the array itself?
            // Wait, if flat is [1,2,3] and shape is [3], we return [1,2,3].
            // If flat is [1,2,3,4] and shape is [2,2].
            // dim=2. shape=[2].
            // We need to split flat into 2 chunks of size 2.
        }

        $chunkSize = (int) (count($flat) / $dim);
        $reshaped = [];
        for ($i = 0; $i < $dim; $i++) {
            $slice = array_slice($flat, $i * $chunkSize, $chunkSize);
            $reshaped[] = $this->reshape($slice, $shape);
        }
        return $reshaped;
    }

    public function updateParams(OptimizerInterface $optimizer, int $layerIndex): void {
        // No params
    }

    public function getParams(): array {
        return [];
    }

    public function setParams(array $params): void {
        // No params
    }

    public function getOutputShape(array $inputShape): array {
        // Calculate total elements per sample
        $total = 1;
        // inputShape[0] is batch size usually, but here inputShape passed might be (batch, d1, d2...) or just (d1, d2...)?
        // LayerInterface::getOutputShape usually takes (batch, d1, d2...).
        // So we preserve batch size.
        $batch = $inputShape[0];
        for ($i = 1; $i < count($inputShape); $i++) {
            $total *= $inputShape[$i];
        }
        return [$batch, $total];
    }
}
