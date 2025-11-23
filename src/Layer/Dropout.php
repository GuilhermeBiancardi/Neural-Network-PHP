<?php
declare (strict_types = 1);

namespace NeuralNetwork\Layer;

use NeuralNetwork\Optimizer\OptimizerInterface;

class Dropout implements LayerInterface {
    private float $rate;
    private array $mask = [];

    public function __construct(float $rate = 0.5) {
        $this->rate = max(0.0, min(1.0, $rate));
    }

    public function forward(array $input, bool $training = false): array {
        if (!$training || $this->rate <= 0.0 || $this->rate >= 1.0) {
            return $input;
        }

        $this->mask = [];
        $keepProb = 1.0 - $this->rate;
        $scale = 1.0 / $keepProb; // Inverted dropout

        $output = [];
        foreach ($input as $row) {
            $maskRow = [];
            $outRow = [];
            foreach ($row as $val) {
                // Handle multi-dimensional input recursively?
                // Usually Dropout is applied to Flattened or Dense output (1D per sample) or Feature Maps (2D/3D).
                // Let's assume input is array of numbers or array of arrays.
                // If input is 4D (Conv2D output), we drop entire features or elements?
                // Standard Dropout drops individual elements.
                // SpatialDropout drops entire feature maps.
                // Let's implement element-wise dropout for any shape.
                // But recursion is slow.
                // Let's assume input is 2D (batch, features) for now as it's most common after Dense.
                // If input is 4D, we need recursion.

                // Let's use a recursive helper.
                // But wait, forward() signature says array.
            }
        }

        // Recursive implementation for any shape
        return $this->applyDropout($input, $training);
    }

    private function applyDropout(array $input, bool $training): array {
        $output = [];
        $mask = [];
        $keepProb = 1.0 - $this->rate;
        $scale = 1.0 / $keepProb;

        foreach ($input as $key => $val) {
            if (is_array($val)) {
                [$outSub, $maskSub] = $this->applyDropoutRecursive($val, $scale, $keepProb);
                $output[$key] = $outSub;
                $mask[$key] = $maskSub;
            } else {
                // Should not happen at top level if batch
            }
        }
        $this->mask = $mask;
        return $output;
    }

    private function applyDropoutRecursive(array $input, float $scale, float $keepProb): array {
        $output = [];
        $mask = [];
        foreach ($input as $key => $val) {
            if (is_array($val)) {
                [$outSub, $maskSub] = $this->applyDropoutRecursive($val, $scale, $keepProb);
                $output[$key] = $outSub;
                $mask[$key] = $maskSub;
            } else {
                if ((mt_rand() / mt_getrandmax()) < $keepProb) {
                    $output[$key] = $val * $scale;
                    $mask[$key] = $scale;
                } else {
                    $output[$key] = 0.0;
                    $mask[$key] = 0.0;
                }
            }
        }
        return [$output, $mask];
    }

    public function backward(array $outputGradient): array {
        // Apply mask to gradient
        return $this->applyMask($outputGradient, $this->mask);
    }

    private function applyMask(array $grad, array $mask): array {
        $result = [];
        foreach ($grad as $key => $val) {
            if (is_array($val)) {
                $result[$key] = $this->applyMask($val, $mask[$key]);
            } else {
                $result[$key] = $val * $mask[$key];
            }
        }
        return $result;
    }

    public function updateParams(OptimizerInterface $optimizer, int $layerIndex): void {
        // No params
    }

    public function getParams(): array {
        return ['rate' => $this->rate];
    }

    public function setParams(array $params): void {
        $this->rate = $params['rate'];
    }

    public function getOutputShape(array $inputShape): array {
        return $inputShape;
    }
}
