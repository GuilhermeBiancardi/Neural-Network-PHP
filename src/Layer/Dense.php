<?php
declare (strict_types = 1);

namespace NeuralNetwork\Layer;

use NeuralNetwork\Activation\ActivationInterface;
use NeuralNetwork\Activation\ELU;
use NeuralNetwork\Activation\Gelu;
use NeuralNetwork\Activation\HardSigmoid;
use NeuralNetwork\Activation\HardSwish;
use NeuralNetwork\Activation\HardTanh;
use NeuralNetwork\Activation\LeakyRelu;
use NeuralNetwork\Activation\Linear;
use NeuralNetwork\Activation\Relu;
use NeuralNetwork\Activation\Sigmoid;
use NeuralNetwork\Activation\Softmax;
use NeuralNetwork\Activation\Tanh;
use NeuralNetwork\Helper\Initializer;
use NeuralNetwork\Helper\Matrix;
use NeuralNetwork\Optimizer\OptimizerInterface;

class Dense implements LayerInterface {
    private int $inputSize;
    private int $outputSize;
    private ActivationInterface $activation;

    /** @var array<int, array<int, float>> */
    private array $weights = [];
    /** @var array<int, float> */
    private array $biases = [];

    // Cache for backward pass
    private array $input = [];
    private array $output = []; // Z (before activation)
    private array $activatedOutput = []; // A

    // Gradients
    private array $dW = [];
    private array $db = [];

    public function __construct(int $inputSize, int $outputSize, $activation = null, string $init = 'xavier') {
        $this->inputSize = $inputSize;
        $this->outputSize = $outputSize;

        if (is_string($activation)) {
            $map = [
                'relu' => Relu::class,
                'elu' => ELU::class,
                'leaky-relu' => LeakyRelu::class,
                'sigmoid' => Sigmoid::class,
                'tanh' => Tanh::class,
                'softmax' => Softmax::class,
                'linear' => Linear::class,
                'gelu' => Gelu::class,
                'hardsigmoid' => HardSigmoid::class,
                'hardswish' => HardSwish::class,
                'hardtanh' => HardTanh::class,
            ];
            $class = $map[strtolower($activation)] ?? null;
            if ($class === null) {
                throw new \InvalidArgumentException("Unsupported activation: $activation");
            }
            $this->activation = new $class();
        } else {
            $this->activation = $activation ?? new Sigmoid();
        }

        $this->initParams($init);
    }

    private function initParams(string $method): void {
        $this->weights = [];
        $this->biases = [];

        for ($j = 0; $j < $this->outputSize; $j++) {
            $this->weights[$j] = [];
            for ($k = 0; $k < $this->inputSize; $k++) {
                if ($method === 'he' || $this->activation instanceof Relu) {
                    $this->weights[$j][$k] = Initializer::he($this->inputSize);
                } else {
                    $this->weights[$j][$k] = Initializer::xavier($this->inputSize, $this->outputSize);
                }
            }
            $this->biases[$j] = 0.0;
        }
    }

    public function forward(array $input, bool $training = false): array {
        // Input: (batch_size x input_size)
        $this->input = Matrix::transpose($input); // (input_size x batch_size)

        $z = Matrix::multiply($this->weights, $this->input); // (output_size x batch_size)

        // Add bias
        $batchSize = count($z[0]);
        for ($r = 0; $r < count($z); $r++) {
            for ($c = 0; $c < $batchSize; $c++) {
                $z[$r][$c] += $this->biases[$r];
            }
        }

        $this->output = $z;
        $this->activatedOutput = $this->activation->activate($z);
        return Matrix::transpose($this->activatedOutput);
    }

    public function backward(array $outputGradient): array {
        // outputGradient: (batch_size x output_size)
        $outputGradientT = Matrix::transpose($outputGradient); // (output_size x batch_size)

        $derivative = $this->activation->derivative($this->output);
        $dZ = Matrix::hadamard($outputGradientT, $derivative);

        // dW = dZ * input^T
        $inputT = Matrix::transpose($this->input);
        $this->dW = Matrix::multiply($dZ, $inputT);

        // db = mean over batch
        $batchSize = count($dZ[0]);
        $this->db = [];
        for ($r = 0; $r < count($dZ); $r++) {
            $sum = 0.0;
            foreach ($dZ[$r] as $val) {
                $sum += $val;
            }
            $this->db[$r] = $sum / $batchSize;
        }

        // Average dW over batch
        for ($r = 0; $r < count($this->dW); $r++) {
            for ($c = 0; $c < count($this->dW[0]); $c++) {
                $this->dW[$r][$c] /= $batchSize;
            }
        }

        // Gradient w.r.t input
        $weightsT = Matrix::transpose($this->weights);
        $dInput = Matrix::multiply($weightsT, $dZ);
        return Matrix::transpose($dInput);
    }

    public function updateParams(OptimizerInterface $optimizer, int $layerIndex): void {
        $optimizer->update($layerIndex, $this->weights, $this->biases, $this->dW, $this->db);
    }

    public function getParams(): array {
        return [
            'weights' => $this->weights,
            'biases' => $this->biases,
            'activation' => get_class($this->activation),
            'input_size' => $this->inputSize,
            'output_size' => $this->outputSize,
        ];
    }

    public function setParams(array $params): void {
        $this->weights = $params['weights'];
        $this->biases = $params['biases'];
        $this->inputSize = $params['input_size'];
        $this->outputSize = $params['output_size'];

        // Restore activation function
        $activationClass = $params['activation'];
        $this->activation = new $activationClass();
    }

    public function getOutputShape(array $inputShape): array {
        // Input shape: [batch_size, input_size]
        return [$inputShape[0], $this->outputSize];
    }
}
?>
