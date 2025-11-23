<?php
declare (strict_types = 1);

namespace NeuralNetwork\Helper;

class Initializer {
    public static function xavier(int $inputSize, int $outputSize): float {
        $limit = sqrt(6 / ($inputSize + $outputSize));
        return (mt_rand() / mt_getrandmax()) * 2 * $limit - $limit;
    }

    public static function he(int $inputSize): float {
        $std = sqrt(2.0 / $inputSize);
        // Box-Muller transform for normal distribution
        $u1 = mt_rand() / mt_getrandmax();
        $u2 = mt_rand() / mt_getrandmax();
        $z = sqrt(-2.0 * log($u1)) * cos(2.0 * M_PI * $u2);
        return $z * $std;
    }
}
