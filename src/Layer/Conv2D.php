<?php
declare (strict_types = 1);

namespace NeuralNetwork\Layer;

use NeuralNetwork\Optimizer\OptimizerInterface;

class Conv2D implements LayerInterface {
    private int $inChannels;
    private int $outChannels;
    private int $kernelSize;
    private int $stride;
    private int $padding;

    // [out_channels][in_channels][kernel_size][kernel_size]
    private array $filters = [];
    // [out_channels]
    private array $biases = [];

    // Cache
    private array $input = [];

    // Gradients
    private array $dFilters = [];
    private array $dBias = [];

    public function __construct(int $inChannels, int $outChannels, int $kernelSize, int $stride = 1, int $padding = 0) {
        $this->inChannels = $inChannels;
        $this->outChannels = $outChannels;
        $this->kernelSize = $kernelSize;
        $this->stride = $stride;
        $this->padding = $padding;

        // Initialize parameters
        $this->initializeParams();
    }

    private function initializeParams(): void {
        $scale = sqrt(2.0 / ($this->inChannels * $this->kernelSize * $this->kernelSize));

        for ($i = 0; $i < $this->outChannels; $i++) {
            $this->biases[$i] = 0.0;
            for ($j = 0; $j < $this->inChannels; $j++) {
                for ($k = 0; $k < $this->kernelSize; $k++) {
                    for ($l = 0; $l < $this->kernelSize; $l++) {
                        // He initialization
                        $this->filters[$i][$j][$k][$l] = (rand(0, 1000) / 1000.0) * $scale * 2 - $scale;
                    }
                }
            }
        }
    }

    public function forward(array $input, bool $training = false): array {
        // Input: [batch, in_channels, height, width]
        $this->input = $input;
        $batchSize = count($input);
        $inHeight = count($input[0][0]);
        $inWidth = count($input[0][0][0]);

        $outHeight = (int) (($inHeight + 2 * $this->padding - $this->kernelSize) / $this->stride) + 1;
        $outWidth = (int) (($inWidth + 2 * $this->padding - $this->kernelSize) / $this->stride) + 1;

        $output = [];

        for ($b = 0; $b < $batchSize; $b++) {
            $sampleOutput = [];
            for ($o = 0; $o < $this->outChannels; $o++) {
                $featureMap = [];
                for ($h = 0; $h < $outHeight; $h++) {
                    $row = [];
                    for ($w = 0; $w < $outWidth; $w++) {
                        $sum = $this->biases[$o];

                        $hStart = $h * $this->stride - $this->padding;
                        $wStart = $w * $this->stride - $this->padding;

                        for ($c = 0; $c < $this->inChannels; $c++) {
                            for ($kh = 0; $kh < $this->kernelSize; $kh++) {
                                for ($kw = 0; $kw < $this->kernelSize; $kw++) {
                                    $inH = $hStart + $kh;
                                    $inW = $wStart + $kw;

                                    if ($inH >= 0 && $inH < $inHeight && $inW >= 0 && $inW < $inWidth) {
                                        $sum += $input[$b][$c][$inH][$inW] * $this->filters[$o][$c][$kh][$kw];
                                    }
                                }
                            }
                        }
                        $row[] = $sum;
                    }
                    $featureMap[] = $row;
                }
                $sampleOutput[] = $featureMap;
            }
            $output[] = $sampleOutput;
        }

        return $output;
    }

    public function backward(array $outputGradient): array {
        // outputGradient: [batch, out_channels, out_height, out_width]
        $batchSize = count($outputGradient);
        $outHeight = count($outputGradient[0][0]);
        $outWidth = count($outputGradient[0][0][0]);

        $inHeight = count($this->input[0][0]);
        $inWidth = count($this->input[0][0][0]);

        // Initialize gradients
        $this->dFilters = []; // Should be zeroed out first
        // Helper to create zero array of shape filters
        for ($i = 0; $i < $this->outChannels; $i++) {
            for ($j = 0; $j < $this->inChannels; $j++) {
                for ($k = 0; $k < $this->kernelSize; $k++) {
                    for ($l = 0; $l < $this->kernelSize; $l++) {
                        $this->dFilters[$i][$j][$k][$l] = 0.0;
                    }
                }
            }
            $this->dBias[$i] = 0.0;
        }

        $dInput = []; // [batch, in_channels, in_height, in_width]
        // Initialize dInput with zeros
        for ($b = 0; $b < $batchSize; $b++) {
            for ($c = 0; $c < $this->inChannels; $c++) {
                for ($h = 0; $h < $inHeight; $h++) {
                    for ($w = 0; $w < $inWidth; $w++) {
                        $dInput[$b][$c][$h][$w] = 0.0;
                    }
                }
            }
        }

        for ($b = 0; $b < $batchSize; $b++) {
            for ($o = 0; $o < $this->outChannels; $o++) {
                for ($h = 0; $h < $outHeight; $h++) {
                    for ($w = 0; $w < $outWidth; $w++) {
                        $grad = $outputGradient[$b][$o][$h][$w];
                        $this->dBias[$o] += $grad;

                        $hStart = $h * $this->stride - $this->padding;
                        $wStart = $w * $this->stride - $this->padding;

                        for ($c = 0; $c < $this->inChannels; $c++) {
                            for ($kh = 0; $kh < $this->kernelSize; $kh++) {
                                for ($kw = 0; $kw < $this->kernelSize; $kw++) {
                                    $inH = $hStart + $kh;
                                    $inW = $wStart + $kw;

                                    if ($inH >= 0 && $inH < $inHeight && $inW >= 0 && $inW < $inWidth) {
                                        // dFilter += input * grad
                                        $this->dFilters[$o][$c][$kh][$kw] += $this->input[$b][$c][$inH][$inW] * $grad;

                                        // dInput += filter * grad
                                        $dInput[$b][$c][$inH][$inW] += $this->filters[$o][$c][$kh][$kw] * $grad;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Average gradients over batch
        for ($i = 0; $i < $this->outChannels; $i++) {
            $this->dBias[$i] /= $batchSize;
            for ($j = 0; $j < $this->inChannels; $j++) {
                for ($k = 0; $k < $this->kernelSize; $k++) {
                    for ($l = 0; $l < $this->kernelSize; $l++) {
                        $this->dFilters[$i][$j][$k][$l] /= $batchSize;
                    }
                }
            }
        }

        return $dInput;
    }

    public function updateParams(OptimizerInterface $optimizer, int $layerIndex): void {
        // Optimizer expects 2D weights usually.
        // We need to flatten filters to 2D or modify optimizer.
        // Or we can just iterate and update manually if optimizer allows?
        // The optimizer interface is: update($layerIndex, &$weights, &$biases, $dW, $db)
        // It assumes weights is array<array<float>>.
        // Our filters is 4D.
        // This is a problem. The existing Optimizer is designed for Dense layers (2D matrices).

        // Solution: Flatten the filters to 2D [out_channels, in_channels * k * k] for the optimizer,
        // then reshape back.

        $flatFilters = [];
        $flatDFilters = [];

        $fanIn = $this->inChannels * $this->kernelSize * $this->kernelSize;

        for ($i = 0; $i < $this->outChannels; $i++) {
            $flatRow = [];
            $flatDRow = [];
            for ($j = 0; $j < $this->inChannels; $j++) {
                for ($k = 0; $k < $this->kernelSize; $k++) {
                    for ($l = 0; $l < $this->kernelSize; $l++) {
                        $flatRow[] = $this->filters[$i][$j][$k][$l];
                        $flatDRow[] = $this->dFilters[$i][$j][$k][$l];
                    }
                }
            }
            $flatFilters[] = $flatRow;
            $flatDFilters[] = $flatDRow;
        }

        $optimizer->update($layerIndex, $flatFilters, $this->biases, $flatDFilters, $this->dBias);

        // Reshape back
        for ($i = 0; $i < $this->outChannels; $i++) {
            $idx = 0;
            for ($j = 0; $j < $this->inChannels; $j++) {
                for ($k = 0; $k < $this->kernelSize; $k++) {
                    for ($l = 0; $l < $this->kernelSize; $l++) {
                        $this->filters[$i][$j][$k][$l] = $flatFilters[$i][$idx];
                        $idx++;
                    }
                }
            }
        }
    }

    public function getParams(): array {
        return [
            'filters' => $this->filters,
            'biases' => $this->biases,
            'in_channels' => $this->inChannels,
            'out_channels' => $this->outChannels,
            'kernel_size' => $this->kernelSize,
            'stride' => $this->stride,
            'padding' => $this->padding,
        ];
    }

    public function setParams(array $params): void {
        $this->filters = $params['filters'];
        $this->biases = $params['biases'];
    }

    public function getOutputShape(array $inputShape): array {
        // inputShape: [batch, in_channels, height, width]
        $inHeight = $inputShape[2];
        $inWidth = $inputShape[3];

        $outHeight = (int) (($inHeight + 2 * $this->padding - $this->kernelSize) / $this->stride) + 1;
        $outWidth = (int) (($inWidth + 2 * $this->padding - $this->kernelSize) / $this->stride) + 1;

        return [$inputShape[0], $this->outChannels, $outHeight, $outWidth];
    }
}
