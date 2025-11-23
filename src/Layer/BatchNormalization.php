<?php
declare (strict_types = 1);

namespace NeuralNetwork\Layer;

use NeuralNetwork\Optimizer\OptimizerInterface;

class BatchNormalization implements LayerInterface {
    private int $numFeatures;
    private float $epsilon = 1e-8;
    private float $momentum = 0.9;

    // Learnable parameters
    private array $gamma; // Scale
    private array $beta; // Shift

    // Running stats for inference
    private array $runningMean;
    private array $runningVar;

    // Cache for backward
    private array $input;
    private array $xCentered;
    private array $xNorm;
    private array $mean;
    private array $var;
    private array $std;

    // Gradients
    private array $dGamma;
    private array $dBeta;

    public function __construct(int $numFeatures, float $momentum = 0.9) {
        $this->numFeatures = $numFeatures;
        $this->momentum = $momentum;

        $this->gamma = array_fill(0, $numFeatures, 1.0);
        $this->beta = array_fill(0, $numFeatures, 0.0);
        $this->runningMean = array_fill(0, $numFeatures, 0.0);
        $this->runningVar = array_fill(0, $numFeatures, 1.0);
    }

    public function forward(array $input, bool $training = false): array {
        // Input: [batch_size, num_features]
        $this->input = $input;
        $batchSize = count($input);

        if ($training) {
            // Calculate mean and var per feature
            $mean = array_fill(0, $this->numFeatures, 0.0);
            $var = array_fill(0, $this->numFeatures, 0.0);

            foreach ($input as $row) {
                for ($j = 0; $j < $this->numFeatures; $j++) {
                    $mean[$j] += $row[$j];
                }
            }
            for ($j = 0; $j < $this->numFeatures; $j++) {
                $mean[$j] /= $batchSize;
            }

            foreach ($input as $row) {
                for ($j = 0; $j < $this->numFeatures; $j++) {
                    $var[$j] += ($row[$j] - $mean[$j]) ** 2;
                }
            }
            for ($j = 0; $j < $this->numFeatures; $j++) {
                $var[$j] /= $batchSize;
            }

            $this->mean = $mean;
            $this->var = $var;

            // Update running stats
            for ($j = 0; $j < $this->numFeatures; $j++) {
                $this->runningMean[$j] = $this->momentum * $this->runningMean[$j] + (1 - $this->momentum) * $mean[$j];
                $this->runningVar[$j] = $this->momentum * $this->runningVar[$j] + (1 - $this->momentum) * $var[$j];
            }
        } else {
            $mean = $this->runningMean;
            $var = $this->runningVar;
        }

        // Normalize
        $output = [];
        $this->xCentered = [];
        $this->xNorm = [];
        $this->std = [];

        for ($j = 0; $j < $this->numFeatures; $j++) {
            $this->std[$j] = sqrt($var[$j] + $this->epsilon);
        }

        for ($i = 0; $i < $batchSize; $i++) {
            $row = [];
            $xCenteredRow = [];
            $xNormRow = [];
            for ($j = 0; $j < $this->numFeatures; $j++) {
                $xc = $input[$i][$j] - $mean[$j];
                $xn = $xc / $this->std[$j];
                $y = $this->gamma[$j] * $xn + $this->beta[$j];

                $row[$j] = $y;
                $xCenteredRow[$j] = $xc;
                $xNormRow[$j] = $xn;
            }
            $output[] = $row;
            $this->xCentered[] = $xCenteredRow;
            $this->xNorm[] = $xNormRow;
        }

        return $output;
    }

    public function backward(array $outputGradient): array {
        // outputGradient: [batch_size, num_features]
        $batchSize = count($outputGradient);
        $dGamma = array_fill(0, $this->numFeatures, 0.0);
        $dBeta = array_fill(0, $this->numFeatures, 0.0);
        $dxNorm = [];

        for ($i = 0; $i < $batchSize; $i++) {
            $dxNormRow = [];
            for ($j = 0; $j < $this->numFeatures; $j++) {
                $dout = $outputGradient[$i][$j];
                $dGamma[$j] += $dout * $this->xNorm[$i][$j];
                $dBeta[$j] += $dout;
                $dxNormRow[$j] = $dout * $this->gamma[$j];
            }
            $dxNorm[] = $dxNormRow;
        }

        $this->dGamma = array_map(fn($v) => $v / $batchSize, $dGamma); // Average? Or Sum?
        // Standard BN backprop usually sums gradients for params.
        // But if we use mean loss, we might need to average.
        // Let's stick to sum for params as standard, but optimizer might expect average?
        // In Dense layer we averaged dW. Let's average here too to be consistent with "batch average gradient".
        $this->dGamma = array_map(fn($v) => $v / $batchSize, $dGamma);
        $this->dBeta = array_map(fn($v) => $v / $batchSize, $dBeta);

        // Calculate dx
        $dx = [];
        for ($i = 0; $i < $batchSize; $i++) {
            $row = [];
            for ($j = 0; $j < $this->numFeatures; $j++) {
                // Complex BN gradient formula
                // dx = (1/m) * (1/std) * ( m*dxNorm - sum(dxNorm) - xNorm * sum(dxNorm * xNorm) )

                // We need sums over batch for each feature
                // Let's precalculate sums
                // This is inefficient inside the loop.
                // Let's move out.
                $row[$j] = 0; // Placeholder
            }
            $dx[] = $row;
        }

        // Precalc sums
        $sumDxNorm = array_fill(0, $this->numFeatures, 0.0);
        $sumDxNormXNorm = array_fill(0, $this->numFeatures, 0.0);

        for ($i = 0; $i < $batchSize; $i++) {
            for ($j = 0; $j < $this->numFeatures; $j++) {
                $sumDxNorm[$j] += $dxNorm[$i][$j];
                $sumDxNormXNorm[$j] += $dxNorm[$i][$j] * $this->xNorm[$i][$j];
            }
        }

        for ($i = 0; $i < $batchSize; $i++) {
            for ($j = 0; $j < $this->numFeatures; $j++) {
                $stdInv = 1.0 / $this->std[$j];
                $term1 = $batchSize * $dxNorm[$i][$j];
                $term2 = $sumDxNorm[$j];
                $term3 = $this->xNorm[$i][$j] * $sumDxNormXNorm[$j];

                $dx[$i][$j] = ($stdInv / $batchSize) * ($term1 - $term2 - $term3);
            }
        }

        return $dx;
    }

    public function updateParams(OptimizerInterface $optimizer, int $layerIndex): void {
        // We need to adapt Optimizer to handle 1D arrays (gamma/beta) if it expects 2D weights.
        // The current OptimizerInterface::update takes &$weights (2D), &$biases (1D).
        // We can treat Gamma as Biases-like (1D) and Beta as Biases-like (1D).
        // But the signature is fixed: update(..., weights, biases, ...).
        // We can pass Gamma as "Biases" and Beta as... wait.
        // We might need a custom update for BN or trick the optimizer.
        // Or pass Gamma as a 1-row matrix for weights?

        $gammaMatrix = [$this->gamma]; // 1 x features
        $dGammaMatrix = [$this->dGamma];

        // We can pass Gamma as weights (1 row) and Beta as biases.
        $optimizer->update($layerIndex, $gammaMatrix, $this->beta, $dGammaMatrix, $this->dBeta);

        $this->gamma = $gammaMatrix[0];
    }

    public function getParams(): array {
        return [
            'gamma' => $this->gamma,
            'beta' => $this->beta,
            'running_mean' => $this->runningMean,
            'running_var' => $this->runningVar,
        ];
    }

    public function setParams(array $params): void {
        $this->gamma = $params['gamma'];
        $this->beta = $params['beta'];
        $this->runningMean = $params['running_mean'];
        $this->runningVar = $params['running_var'];
    }

    public function getOutputShape(array $inputShape): array {
        return $inputShape;
    }
}
