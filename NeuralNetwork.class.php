<?php
declare (strict_types = 1);

// Load Composer autoloader if available (for Rindow, etc.)
if (file_exists(__DIR__ . '/vendor/autoload.php')) {
    require_once __DIR__ . '/vendor/autoload.php';
}

require_once __DIR__ . '/src/Helper/Matrix.php';
require_once __DIR__ . '/src/Helper/Initializer.php';
require_once __DIR__ . '/src/Activation/ActivationInterface.php';
require_once __DIR__ . '/src/Activation/Sigmoid.php';
require_once __DIR__ . '/src/Activation/Relu.php';
require_once __DIR__ . '/src/Activation/Tanh.php';
require_once __DIR__ . '/src/Activation/Softmax.php';
require_once __DIR__ . '/src/Activation/Linear.php';
require_once __DIR__ . '/src/Activation/LeakyRelu.php';
require_once __DIR__ . '/src/Activation/ELU.php';
require_once __DIR__ . '/src/Loss/LossFunctionInterface.php';
require_once __DIR__ . '/src/Loss/MSE.php';
require_once __DIR__ . '/src/Loss/CrossEntropy.php';
require_once __DIR__ . '/src/Optimizer/OptimizerInterface.php';
require_once __DIR__ . '/src/Optimizer/SGD.php';
require_once __DIR__ . '/src/Optimizer/Adam.php';
require_once __DIR__ . '/src/Optimizer/AdamW.php';
require_once __DIR__ . '/src/Optimizer/RMSProp.php';
require_once __DIR__ . '/src/Layer/LayerInterface.php';
require_once __DIR__ . '/src/Layer/Dense.php';
require_once __DIR__ . '/src/Layer/Flatten.php';
require_once __DIR__ . '/src/Layer/BatchNormalization.php';
require_once __DIR__ . '/src/Layer/Conv2D.php';
require_once __DIR__ . '/src/Layer/Dropout.php';

use NeuralNetwork\Layer\LayerInterface;
use NeuralNetwork\Loss\CrossEntropy;
use NeuralNetwork\Loss\LossFunctionInterface;
use NeuralNetwork\Loss\MSE;
use NeuralNetwork\Optimizer\Adam;
use NeuralNetwork\Optimizer\OptimizerInterface;
use NeuralNetwork\Optimizer\RMSProp;
use NeuralNetwork\Optimizer\SGD;

class NeuralNetwork {
    /** @var LayerInterface[] */
    private array $layers;

    private OptimizerInterface $optimizer;
    private LossFunctionInterface $lossFunction;
    private int $batchSize = 32;

    /**
     * @param LayerInterface[] $layers
     */
    public function __construct(array $layers, float $learningRate = 0.01) {
        $this->layers = $layers;
        $this->optimizer = new SGD($learningRate);
        $this->lossFunction = new MSE();
    }

    public function configure(array $options): void {
        $lr = $options['learning_rate'] ?? 0.01;
        $momentum = $options['momentum'] ?? 0.0;
        $weightDecay = $options['weight_decay'] ?? 0.0;
        $clipValue = $options['gradient_clip'] ?? 0.0;

        if (isset($options['optimizer'])) {
            $optName = strtolower($options['optimizer']);
            $this->optimizer = match ($optName) {
                'adam' => new Adam($lr), // Could add clipValue here if we updated Adam
                'adamw' => new \NeuralNetwork\Optimizer\AdamW($lr, 0.9, 0.999, 1e-8, $weightDecay, $clipValue),
                'rmsprop' => new RMSProp($lr),
                'sgd' => new SGD($lr, $momentum, $weightDecay, $clipValue),
                default => new SGD($lr, $momentum, $weightDecay, $clipValue),
            };
        }

        if (isset($options['batch_size'])) {
            $this->batchSize = max(1, (int) $options['batch_size']);
        }

        if (isset($options['loss'])) {
            if ($options['loss'] === 'crossentropy') {
                $this->lossFunction = new CrossEntropy();
            } elseif ($options['loss'] === 'mse') {
                $this->lossFunction = new MSE();
            }
        }
    }

    public function predict(array $input): array {
        // Auto-detect single sample (1D array) vs batch (2D array)
        $isSingleSample = !empty($input) && !is_array($input[0]);

        if ($isSingleSample) {
            $input = [$input];
        }

        $output = $input;
        foreach ($this->layers as $layer) {
            $output = $layer->forward($output, false);
        }

        // If we wrapped it, we should probably unwrap the result to return a single sample output
        // But the previous API returned array.
        // If input was [x, y], output should be [out1, out2].
        // If we return [[out1, out2]], it breaks backward compat.

        if ($isSingleSample) {
            return $output[0];
        }

        return $output;
    }

    public function train(array $inputs, array $targets, int $epochs, int $patience = 0, bool $verbose = false): void {
        $numSamples = count($inputs);
        $bestLoss = INF;
        $patienceCounter = 0;

        for ($e = 0; $e < $epochs; $e++) {
            $indices = range(0, $numSamples - 1);
            shuffle($indices);
            $epochLoss = 0.0;
            $batches = 0;

            for ($start = 0; $start < $numSamples; $start += $this->batchSize) {
                $batchIndices = array_slice($indices, $start, $this->batchSize);

                $batchInputs = [];
                $batchTargets = [];
                foreach ($batchIndices as $idx) {
                    $batchInputs[] = $inputs[$idx];
                    $batchTargets[] = $targets[$idx];
                }

                // Forward
                $output = $batchInputs;
                foreach ($this->layers as $layer) {
                    $output = $layer->forward($output, true);
                }

                // Loss
                $loss = $this->lossFunction->calculate($output, $batchTargets);
                $epochLoss += $loss;
                $batches++;

                // Backward
                $gradient = $this->lossFunction->calculateGradient($output, $batchTargets);

                // Backpropagate through layers in reverse
                for ($i = count($this->layers) - 1; $i >= 0; $i--) {
                    $gradient = $this->layers[$i]->backward($gradient);
                    $this->layers[$i]->updateParams($this->optimizer, $i);
                }
            }

            $epochLoss /= $batches;

            if ($verbose && ($e % 1 == 0)) { // Verbose every epoch
                echo "Epoch $e: Loss = $epochLoss\n";
            }

            // Early Stopping
            if ($patience > 0) {
                if ($epochLoss < $bestLoss) {
                    $bestLoss = $epochLoss;
                    $patienceCounter = 0;
                } else {
                    $patienceCounter++;
                    if ($patienceCounter >= $patience) {
                        if ($verbose) {
                            echo "Early stopping at epoch $e. Best Loss: $bestLoss\n";
                        }
                        break;
                    }
                }
            }
        }
    }

    public function save(string $filepath): void {
        $data = [
            'layers' => [],
            'optimizer' => serialize($this->optimizer), // Serialize optimizer state
            'loss' => serialize($this->lossFunction),
        ];

        foreach ($this->layers as $layer) {
            $data['layers'][] = [
                'class' => get_class($layer),
                'params' => $layer->getParams(),
            ];
        }

        file_put_contents($filepath, serialize($data));
    }

    public static function load(string $filepath): NeuralNetwork {
        if (!file_exists($filepath)) {
            throw new Exception("File not found: $filepath");
        }

        $data = unserialize(file_get_contents($filepath));

        $layers = [];
        foreach ($data['layers'] as $layerData) {
            $className = $layerData['class'];
            // We need to instantiate the layer.
            // This is tricky because constructors vary.
            // But we can instantiate without constructor using Reflection or just assume empty constructor?
            // No, we should probably use Reflection to instantiate without constructor, then setParams.

            $reflector = new ReflectionClass($className);
            $layer = $reflector->newInstanceWithoutConstructor();
            if ($layer instanceof LayerInterface) {
                $layer->setParams($layerData['params']);
                $layers[] = $layer;
            }
        }

        $nn = new NeuralNetwork($layers);
        $nn->optimizer = unserialize($data['optimizer']);
        $nn->lossFunction = unserialize($data['loss']);

        return $nn;
    }
}
