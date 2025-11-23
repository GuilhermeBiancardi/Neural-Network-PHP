<?php
declare (strict_types = 1);

namespace NeuralNetwork\Helper;

use InvalidArgumentException;
use Tensor\Matrix as TensorMatrix;

class Matrix {
    // Backend detection cache
    private static ?string $backend = null;
    private static ?bool $tensorAvailable = null;
    private static ?bool $rindowAvailable = null;
    private static $matrixOperator = null;

    // Configuration
    private static bool $verbose = false;
    private static ?string $preferredBackend = null;

    /**
     * Detect and return the best available backend
     * Priority: Rindow (GPU) -> Tensor (CPU optimized) -> PHP (pure)
     */
    private static function detectBackend(): string {
        if (self::$backend !== null) {
            return self::$backend;
        }

        // Check if user has set a preferred backend
        if (self::$preferredBackend !== null) {
            self::$backend = self::$preferredBackend;
            if (self::$verbose) {
                echo "[Matrix] Using preferred backend: " . self::$backend . "\n";
            }
            return self::$backend;
        }

        // Auto-detect best available backend
        if (self::isRindowAvailable()) {
            self::$backend = 'rindow';
            if (self::$verbose) {
                echo "[Matrix] GPU Acceleration enabled via Rindow Math Matrix (OpenCL)\n";
            }
        } elseif (self::isTensorAvailable()) {
            self::$backend = 'tensor';
            if (self::$verbose) {
                echo "[Matrix] Using Tensor extension (CPU optimized)\n";
            }
        } else {
            self::$backend = 'php';
            if (self::$verbose) {
                echo "[Matrix] Using pure PHP implementation (slowest)\n";
            }
        }

        return self::$backend;
    }

    /**
     * Check if Rindow Math Matrix is available
     */
    public static function isRindowAvailable(): bool {
        if (self::$rindowAvailable === null) {
            self::$rindowAvailable = class_exists('Rindow\\Math\\Matrix\\MatrixOperator');

            if (self::$rindowAvailable) {
                try {
                    // Try to instantiate to verify it works
                    self::getMatrixOperator();
                } catch (\Exception $e) {
                    self::$rindowAvailable = false;
                    if (self::$verbose) {
                        echo "[Matrix] Rindow available but failed to initialize: " . $e->getMessage() . "\n";
                    }
                }
            }
        }
        return self::$rindowAvailable;
    }

    /**
     * Check if Tensor extension is available
     */
    public static function isTensorAvailable(): bool {
        if (self::$tensorAvailable === null) {
            self::$tensorAvailable = extension_loaded('tensor') && class_exists('Tensor\\Matrix');
        }
        return self::$tensorAvailable;
    }

    /**
     * Get or create Rindow MatrixOperator instance
     */
    private static function getMatrixOperator() {
        if (self::$matrixOperator === null) {
            self::$matrixOperator = new \Rindow\Math\Matrix\MatrixOperator();
        }
        return self::$matrixOperator;
    }

    /**
     * Check if GPU acceleration is available
     */
    public static function isGpuAvailable(): bool {
        return self::isRindowAvailable();
    }

    /**
     * Get information about the current backend
     */
    public static function getBackendInfo(): array {
        $backend = self::detectBackend();

        return [
            'backend' => $backend,
            'gpu_enabled' => $backend === 'rindow',
            'rindow_available' => self::isRindowAvailable(),
            'tensor_available' => self::isTensorAvailable(),
            'description' => match ($backend) {
                'rindow' => 'GPU acceleration via Rindow Math Matrix (OpenCL)',
                'tensor' => 'CPU optimization via Tensor extension',
                'php' => 'Pure PHP implementation',
                default => 'Unknown backend'
            },
        ];
    }

    /**
     * Set preferred backend (for testing or forcing specific implementation)
     *
     * @param string|null $backend 'rindow', 'tensor', 'php', or null for auto-detect
     */
    public static function setPreferredBackend(?string $backend): void {
        if ($backend !== null && !in_array($backend, ['rindow', 'tensor', 'php'])) {
            throw new InvalidArgumentException("Invalid backend: $backend. Must be 'rindow', 'tensor', 'php', or null");
        }

        self::$preferredBackend = $backend;
        self::$backend = null; // Reset to force re-detection
    }

    /**
     * Enable or disable verbose logging
     */
    public static function setVerbose(bool $verbose): void {
        self::$verbose = $verbose;
    }

    // ==================== Matrix Operations ====================

    public static function multiply(array $a, array $b): array {
        $backend = self::detectBackend();

        return match ($backend) {
            'rindow' => self::multiplyRindow($a, $b),
            'tensor' => self::multiplyTensor($a, $b),
            default => self::multiplyPHP($a, $b)
        };
    }

    private static function multiplyRindow(array $a, array $b): array {
        $mo = self::getMatrixOperator();
        $ndA = $mo->array($a);
        $ndB = $mo->array($b);

        // Use la()->gemm() for matrix multiplication
        // gemm: C = alpha * A @ B + beta * C
        $m = count($a);
        $n = count($b[0]);
        $k = count($a[0]);

        $result = $mo->zeros([$m, $n]);
        $mo->la()->gemm($ndA, $ndB, 1.0, 0.0, $result);

        return $result->toArray();
    }

    private static function multiplyTensor(array $a, array $b): array {
        try {
            $tA = TensorMatrix::quick($a);
            $tB = TensorMatrix::quick($b);
            return $tA->matmul($tB)->asArray();
        } catch (\Exception $e) {
            throw $e;
        }
    }

    private static function multiplyPHP(array $a, array $b): array {
        $r1 = count($a);
        $c1 = count($a[0]);
        $r2 = count($b);
        $c2 = count($b[0]);

        if ($c1 !== $r2) {
            throw new InvalidArgumentException("Matrix dimensions mismatch for multiplication: $c1 != $r2");
        }

        // Optimization: Transpose B to access it row-wise (better cache locality)
        $bT = self::transpose($b);

        $result = [];
        for ($i = 0; $i < $r1; $i++) {
            $rowA = $a[$i];
            $resultRow = [];
            for ($j = 0; $j < $c2; $j++) {
                $rowB = $bT[$j];
                $sum = 0.0;
                for ($k = 0; $k < $c1; $k++) {
                    $sum += $rowA[$k] * $rowB[$k];
                }
                $resultRow[] = $sum;
            }
            $result[] = $resultRow;
        }
        return $result;
    }

    public static function add(array $a, array $b): array {
        $backend = self::detectBackend();

        return match ($backend) {
            'rindow' => self::addRindow($a, $b),
            'tensor' => self::addTensor($a, $b),
            default => self::addPHP($a, $b)
        };
    }

    private static function addRindow(array $a, array $b): array {
        // Rindow's add is complex, use PHP fallback for element-wise ops
        return self::addPHP($a, $b);
    }

    private static function addTensor(array $a, array $b): array {
        $tA = TensorMatrix::quick($a);
        $tB = TensorMatrix::quick($b);
        return $tA->add($tB)->asArray();
    }

    private static function addPHP(array $a, array $b): array {
        self::checkDimensions($a, $b, 'addition');
        $result = [];
        $rows = count($a);
        $cols = count($a[0]);

        for ($i = 0; $i < $rows; $i++) {
            $row = [];
            for ($j = 0; $j < $cols; $j++) {
                $row[] = $a[$i][$j] + $b[$i][$j];
            }
            $result[] = $row;
        }
        return $result;
    }

    public static function subtract(array $a, array $b): array {
        $backend = self::detectBackend();

        return match ($backend) {
            'rindow' => self::subtractRindow($a, $b),
            'tensor' => self::subtractTensor($a, $b),
            default => self::subtractPHP($a, $b)
        };
    }

    private static function subtractRindow(array $a, array $b): array {
        // Rindow's sub is complex, use PHP fallback for element-wise ops
        return self::subtractPHP($a, $b);
    }

    private static function subtractTensor(array $a, array $b): array {
        $tA = TensorMatrix::quick($a);
        $tB = TensorMatrix::quick($b);
        return $tA->subtract($tB)->asArray();
    }

    private static function subtractPHP(array $a, array $b): array {
        self::checkDimensions($a, $b, 'subtraction');
        $result = [];
        $rows = count($a);
        $cols = count($a[0]);

        for ($i = 0; $i < $rows; $i++) {
            $row = [];
            for ($j = 0; $j < $cols; $j++) {
                $row[] = $a[$i][$j] - $b[$i][$j];
            }
            $result[] = $row;
        }
        return $result;
    }

    public static function hadamard(array $a, array $b): array {
        $backend = self::detectBackend();

        return match ($backend) {
            'rindow' => self::hadamardRindow($a, $b),
            'tensor' => self::hadamardTensor($a, $b),
            default => self::hadamardPHP($a, $b)
        };
    }

    private static function hadamardRindow(array $a, array $b): array {
        // Rindow's element-wise multiply is complex, use PHP fallback
        return self::hadamardPHP($a, $b);
    }

    private static function hadamardTensor(array $a, array $b): array {
        $tA = TensorMatrix::quick($a);
        $tB = TensorMatrix::quick($b);
        return $tA->multiply($tB)->asArray();
    }

    private static function hadamardPHP(array $a, array $b): array {
        self::checkDimensions($a, $b, 'Hadamard product');
        $result = [];
        $rows = count($a);
        $cols = count($a[0]);

        for ($i = 0; $i < $rows; $i++) {
            $row = [];
            for ($j = 0; $j < $cols; $j++) {
                $row[] = $a[$i][$j] * $b[$i][$j];
            }
            $result[] = $row;
        }
        return $result;
    }

    public static function transpose(array $m): array {
        $backend = self::detectBackend();

        return match ($backend) {
            'rindow' => self::transposeRindow($m),
            'tensor' => self::transposeTensor($m),
            default => self::transposePHP($m)
        };
    }

    private static function transposeRindow(array $m): array {
        $mo = self::getMatrixOperator();
        $ndM = $mo->array($m);
        return $mo->transpose($ndM)->toArray();
    }

    private static function transposeTensor(array $m): array {
        $tM = TensorMatrix::quick($m);
        return $tM->transpose()->asArray();
    }

    private static function transposePHP(array $m): array {
        $result = [];
        $rows = count($m);
        $cols = count($m[0]);

        for ($j = 0; $j < $cols; $j++) {
            $row = [];
            for ($i = 0; $i < $rows; $i++) {
                $row[] = $m[$i][$j];
            }
            $result[] = $row;
        }
        return $result;
    }

    public static function clip(array $m, float $min, float $max): array {
        $backend = self::detectBackend();

        return match ($backend) {
            'rindow' => self::clipRindow($m, $min, $max),
            'tensor' => self::clipTensor($m, $min, $max),
            default => self::clipPHP($m, $min, $max)
        };
    }

    private static function clipRindow(array $m, float $min, float $max): array {
        $mo = self::getMatrixOperator();
        $ndM = $mo->array($m);
        return $mo->clip($ndM, $min, $max)->toArray();
    }

    private static function clipTensor(array $m, float $min, float $max): array {
        $tM = TensorMatrix::quick($m);
        return $tM->clip($min, $max)->asArray();
    }

    private static function clipPHP(array $m, float $min, float $max): array {
        $result = [];
        foreach ($m as $row) {
            $newRow = [];
            foreach ($row as $val) {
                $newRow[] = max($min, min($max, $val));
            }
            $result[] = $newRow;
        }
        return $result;
    }

    public static function scalarMultiply(array $matrix, float $scalar): array {
        $backend = self::detectBackend();

        return match ($backend) {
            'rindow' => self::scalarMultiplyRindow($matrix, $scalar),
            'tensor' => self::scalarMultiplyTensor($matrix, $scalar),
            default => self::scalarMultiplyPHP($matrix, $scalar)
        };
    }

    private static function scalarMultiplyRindow(array $matrix, float $scalar): array {
        // Rindow's scal modifies in-place, use PHP fallback
        return self::scalarMultiplyPHP($matrix, $scalar);
    }

    private static function scalarMultiplyTensor(array $matrix, float $scalar): array {
        $tM = TensorMatrix::quick($matrix);
        return $tM->multiply($scalar)->asArray();
    }

    private static function scalarMultiplyPHP(array $matrix, float $scalar): array {
        $result = [];
        foreach ($matrix as $row) {
            $newRow = [];
            foreach ($row as $val) {
                $newRow[] = $val * $scalar;
            }
            $result[] = $newRow;
        }
        return $result;
    }

    public static function checkDimensions(array $a, array $b, string $operation): void {
        if (count($a) !== count($b) || count($a[0]) !== count($b[0])) {
            throw new InvalidArgumentException("Matrix dimensions mismatch for $operation.");
        }
    }
}
