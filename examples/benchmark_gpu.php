<?php
/**
 * GPU Acceleration Benchmark
 *
 * This script benchmarks the performance of different matrix backends:
 * - Rindow Math Matrix (GPU via OpenCL)
 * - Tensor Extension (CPU optimized)
 * - Pure PHP Implementation
 *
 * Usage: php benchmark_gpu.php
 */

require_once __DIR__ . '/../NeuralNetwork.class.php';

use NeuralNetwork\Helper\Matrix;

// Enable verbose mode to see which backend is being used
Matrix::setVerbose(true);

echo "\n";
echo "╔══════════════════════════════════════════════════════════════╗\n";
echo "║          GPU Acceleration Benchmark Suite                    ║\n";
echo "╚══════════════════════════════════════════════════════════════╝\n";
echo "\n";

// Display backend information
$info = Matrix::getBackendInfo();
echo "Current Backend Configuration:\n";
echo "  Backend: " . $info['backend'] . "\n";
echo "  Description: " . $info['description'] . "\n";
echo "  GPU Enabled: " . ($info['gpu_enabled'] ? 'Yes' : 'No') . "\n";
echo "  Rindow Available: " . ($info['rindow_available'] ? 'Yes' : 'No') . "\n";
echo "  Tensor Available: " . ($info['tensor_available'] ? 'Yes' : 'No') . "\n";
echo "\n";

/**
 * Benchmark a matrix operation
 */
function benchmark(string $name, callable $operation, int $iterations = 10): float {
    // Warm-up
    $operation();

    $times = [];
    for ($i = 0; $i < $iterations; $i++) {
        $start = microtime(true);
        $operation();
        $end = microtime(true);
        $times[] = ($end - $start) * 1000; // Convert to milliseconds
    }

    // Remove outliers (highest and lowest)
    sort($times);
    array_shift($times);
    array_pop($times);

    return array_sum($times) / count($times);
}

/**
 * Generate random matrix
 */
function generateMatrix(int $rows, int $cols): array {
    $matrix = [];
    for ($i = 0; $i < $rows; $i++) {
        $row = [];
        for ($j = 0; $j < $cols; $j++) {
            $row[] = (float) rand(-100, 100) / 10;
        }
        $matrix[] = $row;
    }
    return $matrix;
}

/**
 * Run benchmark for a specific matrix size
 */
function runBenchmark(int $size, string $backend): array {
    Matrix::setPreferredBackend($backend);
    Matrix::setVerbose(false);

    $a = generateMatrix($size, $size);
    $b = generateMatrix($size, $size);

    $results = [];

    // Matrix Multiplication
    $results['multiply'] = benchmark(
        "Matrix Multiply ({$size}x{$size})",
        fn() => Matrix::multiply($a, $b),
        5
    );

    // Matrix Addition
    $results['add'] = benchmark(
        "Matrix Add ({$size}x{$size})",
        fn() => Matrix::add($a, $b),
        10
    );

    // Matrix Transpose
    $results['transpose'] = benchmark(
        "Matrix Transpose ({$size}x{$size})",
        fn() => Matrix::transpose($a),
        10
    );

    // Hadamard Product
    $results['hadamard'] = benchmark(
        "Hadamard Product ({$size}x{$size})",
        fn() => Matrix::hadamard($a, $b),
        10
    );

    return $results;
}

/**
 * Display results table
 */
function displayResults(array $results): void {
    $sizes = array_keys($results);
    $backends = array_keys($results[$sizes[0]]);
    $operations = array_keys($results[$sizes[0]][$backends[0]]);

    foreach ($operations as $op) {
        echo "\n";
        echo "┌─────────────────────────────────────────────────────────────┐\n";
        echo "│ Operation: " . str_pad(ucfirst($op), 47) . "│\n";
        echo "├──────────┬──────────────┬──────────────┬──────────────────┤\n";
        echo "│   Size   │";

        foreach ($backends as $backend) {
            echo " " . str_pad(ucfirst($backend), 12) . " │";
        }
        echo "\n";
        echo "├──────────┼──────────────┼──────────────┼──────────────────┤\n";

        foreach ($sizes as $size) {
            echo "│ " . str_pad($size . "x" . $size, 8) . " │";

            $phpTime = $results[$size]['php'][$op];

            foreach ($backends as $backend) {
                $time = $results[$size][$backend][$op];
                $speedup = $phpTime / $time;

                if ($backend === 'php') {
                    echo " " . str_pad(number_format($time, 2) . " ms", 12) . " │";
                } else {
                    $speedupStr = number_format($speedup, 1) . "x";
                    echo " " . str_pad(number_format($time, 2) . " ms", 8) . " " . str_pad($speedupStr, 3) . "│";
                }
            }
            echo "\n";
        }

        echo "└──────────┴──────────────┴──────────────┴──────────────────┘\n";
    }
}

/**
 * Calculate and display speedup summary
 */
function displaySpeedupSummary(array $results): void {
    echo "\n";
    echo "╔══════════════════════════════════════════════════════════════╗\n";
    echo "║                    Speedup Summary                           ║\n";
    echo "╚══════════════════════════════════════════════════════════════╝\n";
    echo "\n";

    $sizes = array_keys($results);
    $backends = array_keys($results[$sizes[0]]);
    $operations = array_keys($results[$sizes[0]][$backends[0]]);

    foreach ($backends as $backend) {
        if ($backend === 'php') {
            continue;
        }

        echo ucfirst($backend) . " vs PHP:\n";

        foreach ($operations as $op) {
            $speedups = [];
            foreach ($sizes as $size) {
                $phpTime = $results[$size]['php'][$op];
                $backendTime = $results[$size][$backend][$op];
                $speedups[] = $phpTime / $backendTime;
            }

            $avgSpeedup = array_sum($speedups) / count($speedups);
            $maxSpeedup = max($speedups);

            echo "  " . str_pad(ucfirst($op) . ":", 15) .
            "Avg: " . number_format($avgSpeedup, 1) . "x  " .
            "Max: " . number_format($maxSpeedup, 1) . "x\n";
        }
        echo "\n";
    }
}

// ==================== Run Benchmarks ====================

echo "Starting benchmarks...\n";
echo "(This may take a few minutes)\n\n";

$testSizes = [10, 50, 100, 500];
$availableBackends = ['php'];

if (Matrix::isTensorAvailable()) {
    $availableBackends[] = 'tensor';
}

if (Matrix::isRindowAvailable()) {
    $availableBackends[] = 'rindow';
}

$allResults = [];

foreach ($testSizes as $size) {
    echo "Testing {$size}x{$size} matrices...\n";
    $allResults[$size] = [];

    foreach ($availableBackends as $backend) {
        echo "  - Using $backend backend...";
        $allResults[$size][$backend] = runBenchmark($size, $backend);
        echo " ✓\n";
    }
}

echo "\n";
echo "╔══════════════════════════════════════════════════════════════╗\n";
echo "║                    Benchmark Results                         ║\n";
echo "╚══════════════════════════════════════════════════════════════╝\n";

displayResults($allResults);
displaySpeedupSummary($allResults);

// ==================== Neural Network Simulation ====================

echo "\n";
echo "╔══════════════════════════════════════════════════════════════╗\n";
echo "║          Neural Network Training Simulation                  ║\n";
echo "╚══════════════════════════════════════════════════════════════╝\n";
echo "\n";

function simulateNeuralNetworkTraining(string $backend, int $batchSize = 32): float {
    Matrix::setPreferredBackend($backend);
    Matrix::setVerbose(false);

    // Simulate a forward pass
    $input = generateMatrix($batchSize, 784); // 28x28 image flattened
    $weights1 = generateMatrix(784, 128);
    $weights2 = generateMatrix(128, 10);

    $start = microtime(true);

    // Forward pass
    $hidden = Matrix::multiply($input, $weights1);
    $output = Matrix::multiply($hidden, $weights2);

    // Backward pass (simplified)
    $gradOutput = generateMatrix($batchSize, 10);
    $gradHidden = Matrix::multiply($gradOutput, Matrix::transpose($weights2));
    $gradWeights1 = Matrix::multiply(Matrix::transpose($input), $gradHidden);

    $end = microtime(true);

    return ($end - $start) * 1000;
}

echo "Simulating training iteration (batch size: 32):\n\n";

foreach ($availableBackends as $backend) {
    $time = simulateNeuralNetworkTraining($backend);
    echo "  " . str_pad(ucfirst($backend) . ":", 10) . number_format($time, 2) . " ms\n";
}

$phpTime = simulateNeuralNetworkTraining('php');

if (in_array('rindow', $availableBackends)) {
    $rindowTime = simulateNeuralNetworkTraining('rindow');
    $speedup = $phpTime / $rindowTime;
    echo "\n";
    echo "  GPU Speedup: " . number_format($speedup, 1) . "x faster\n";
    echo "  Time saved per iteration: " . number_format($phpTime - $rindowTime, 2) . " ms\n";
    echo "  Time saved per 1000 iterations: " . number_format(($phpTime - $rindowTime) / 1000, 2) . " seconds\n";
}

echo "\n";
echo "╔══════════════════════════════════════════════════════════════╗\n";
echo "║                    Recommendations                           ║\n";
echo "╚══════════════════════════════════════════════════════════════╝\n";
echo "\n";

if (Matrix::isGpuAvailable()) {
    echo " GPU acceleration is ENABLED!\n";
    echo "   Your neural network training will be significantly faster.\n";
    echo "   Recommended for:\n";
    echo "   - Large matrices (>100x100)\n";
    echo "   - Batch processing\n";
    echo "   - Deep neural networks\n";
} elseif (Matrix::isTensorAvailable()) {
    echo " Tensor extension is available (CPU optimized)\n";
    echo "   Consider installing Rindow Math Matrix for GPU acceleration:\n";
    echo "   composer require rindow/rindow-math-matrix\n";
    echo "   See GPU_SETUP.md for installation instructions.\n";
} else {
    echo "  Using pure PHP implementation\n";
    echo "   For better performance, install:\n";
    echo "   1. Tensor extension (CPU optimization)\n";
    echo "   2. Rindow Math Matrix (GPU acceleration)\n";
    echo "   See GPU_SETUP.md for installation instructions.\n";
}

echo "\n";
echo "Benchmark completed!\n";
echo "\n";
