<?php
declare (strict_types = 1);

namespace NeuralNetwork\Helper;

use InvalidArgumentException;
use Tensor\Matrix as TensorMatrix;

class Matrix {
    private static ?bool $tensorAvailable = null;

    public static function isTensorAvailable(): bool {
        if (self::$tensorAvailable === null) {
            self::$tensorAvailable = extension_loaded('tensor') && class_exists('Tensor\Matrix');
        }
        return self::$tensorAvailable;
    }

    public static function multiply(array $a, array $b): array {
        if (self::isTensorAvailable()) {
            try {
                $tA = TensorMatrix::quick($a);
                $tB = TensorMatrix::quick($b);
                return $tA->matmul($tB)->asArray();
            } catch (\Exception $e) {
                // Fallback or rethrow? Let's fallback for robustness if it's a runtime error,
                // but usually we want to know if something failed.
                // However, for now, let's assume if it fails we might want to fallback or just let it bubble.
                // Given the user request "verify if available... if yes use it... if no use as is",
                // we should stick to the happy path. If it crashes, it crashes.
                throw $e;
            }
        }

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
        if (self::isTensorAvailable()) {
            $tA = TensorMatrix::quick($a);
            $tB = TensorMatrix::quick($b);
            return $tA->add($tB)->asArray();
        }

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
        if (self::isTensorAvailable()) {
            $tA = TensorMatrix::quick($a);
            $tB = TensorMatrix::quick($b);
            return $tA->subtract($tB)->asArray();
        }

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
        if (self::isTensorAvailable()) {
            $tA = TensorMatrix::quick($a);
            $tB = TensorMatrix::quick($b);
            return $tA->multiply($tB)->asArray();
        }

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
        if (self::isTensorAvailable()) {
            $tM = TensorMatrix::quick($m);
            return $tM->transpose()->asArray();
        }

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
        if (self::isTensorAvailable()) {
            $tM = TensorMatrix::quick($m);
            return $tM->clip($min, $max)->asArray();
        }

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
        if (self::isTensorAvailable()) {
            $tM = TensorMatrix::quick($matrix);
            return $tM->multiply($scalar)->asArray();
        }

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
