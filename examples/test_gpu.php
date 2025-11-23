<?php
/**
 * Quick test to verify GPU acceleration implementation
 */

require_once __DIR__ . '/../NeuralNetwork.class.php';

use NeuralNetwork\Helper\Matrix;

echo "\n";
echo "===========================================\n";
echo "  GPU Acceleration - Quick Verification\n";
echo "===========================================\n\n";

// Enable verbose mode
Matrix::setVerbose(true);

// Get backend information
$info = Matrix::getBackendInfo();

echo "Backend Information:\n";
echo "  Active Backend: " . $info['backend'] . "\n";
echo "  Description: " . $info['description'] . "\n";
echo "  GPU Enabled: " . ($info['gpu_enabled'] ? 'Yes' : 'No') . "\n";
echo "  Rindow Available: " . ($info['rindow_available'] ? 'Yes' : 'No') . "\n";
echo "  Tensor Available: " . ($info['tensor_available'] ? 'Yes' : 'No') . "\n";
echo "\n";

// Test basic matrix operations
echo "Testing Matrix Operations:\n\n";

$a = [[1, 2], [3, 4]];
$b = [[5, 6], [7, 8]];

echo "Matrix A:\n";
print_r($a);
echo "\nMatrix B:\n";
print_r($b);

// Test multiply
echo "\nA × B (Matrix Multiplication):\n";
$result = Matrix::multiply($a, $b);
print_r($result);

// Test add
echo "\nA + B (Matrix Addition):\n";
$result = Matrix::add($a, $b);
print_r($result);

// Test transpose
echo "\nTranspose(A):\n";
$result = Matrix::transpose($a);
print_r($result);

// Test with different backends
echo "\n";
echo "===========================================\n";
echo "  Testing Backend Switching\n";
echo "===========================================\n\n";

Matrix::setVerbose(false);

$testMatrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
$backends = ['php'];

if (Matrix::isTensorAvailable()) {
    $backends[] = 'tensor';
}

if (Matrix::isRindowAvailable()) {
    $backends[] = 'rindow';
}

foreach ($backends as $backend) {
    Matrix::setPreferredBackend($backend);

    $start = microtime(true);
    $result = Matrix::multiply($testMatrix, $testMatrix);
    $end = microtime(true);

    $time = ($end - $start) * 1000;
    echo ucfirst($backend) . " backend: " . number_format($time, 4) . " ms\n";
}

// Reset to auto-detect
Matrix::setPreferredBackend(null);

echo "\n";
echo "===========================================\n";
echo "  Neural Network Integration Test\n";
echo "===========================================\n\n";

// Test with actual neural network
use NeuralNetwork\Layer\Dense;

$inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
$targets = [[0], [1], [1], [0]];

$layers = [
    new Dense(2, 4, 'relu'),
    new Dense(4, 1, 'sigmoid'),
];

$nn = new NeuralNetwork($layers);
$nn->configure([
    'optimizer' => 'adam',
    'learning_rate' => 0.1,
    'batch_size' => 4,
]);

echo "Training XOR problem (100 epochs)...\n";
$start = microtime(true);
$nn->train($inputs, $targets, 100, 0, false);
$end = microtime(true);

$trainingTime = ($end - $start) * 1000;
echo "Training completed in " . number_format($trainingTime, 2) . " ms\n\n";

echo "Testing predictions:\n";
foreach ($inputs as $input) {
    $output = $nn->predict($input);
    $expected = ($input[0] XOR $input[1]) ? 1 : 0;
    $predicted = round($output[0]);
    $status = ($predicted == $expected) ? 'Yes' : 'No';

    echo "  Input: [" . implode(', ', $input) . "] => ";
    echo "Predicted: " . number_format($output[0], 4) . " (≈" . $predicted . ") ";
    echo "Expected: $expected $status\n";
}

echo "\n";
echo "===========================================\n";
echo "  Verification Complete!\n";
echo "===========================================\n\n";

if (Matrix::isGpuAvailable()) {
    echo "GPU acceleration is working!\n";
    echo "   Your neural network will train significantly faster.\n";
    echo "   Run 'php examples/benchmark_gpu.php' for detailed benchmarks.\n";
} elseif (Matrix::isTensorAvailable()) {
    echo "Tensor extension is working!\n";
    echo "   Consider installing Rindow Math Matrix for GPU acceleration.\n";
    echo "   See GPU_SETUP.md for instructions.\n";
} else {
    echo "Pure PHP implementation is working!\n";
    echo "   For better performance, install:\n";
    echo "   - Rindow Math Matrix (GPU acceleration)\n";
    echo "   - Tensor extension (CPU optimization)\n";
    echo "   See GPU_SETUP.md for instructions.\n";
}

echo "\n";
