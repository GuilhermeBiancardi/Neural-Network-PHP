<?php
declare (strict_types = 1);

namespace NeuralNetwork\Helper;

class Metrics {
    public static function accuracy(array $predictions, array $targets): float {
        $correct = 0;
        $total = count($predictions);

        foreach ($predictions as $i => $pred) {
            // Assume one-hot or probability distribution for classification
            // Or single value for binary classification

            if (is_array($pred)) {
                // Multi-class
                $predClass = self::argmax($pred);
                $targetClass = is_array($targets[$i]) ? self::argmax($targets[$i]) : (int) $targets[$i];
                if ($predClass === $targetClass) {
                    $correct++;
                }
            } else {
                // Binary (sigmoid output)
                $p = $pred >= 0.5 ? 1 : 0;
                $t = $targets[$i] >= 0.5 ? 1 : 0;
                if ($p === $t) {
                    $correct++;
                }
            }
        }

        return $total > 0 ? $correct / $total : 0.0;
    }

    public static function argmax(array $array): int {
        $max = -INF;
        $idx = 0;
        foreach ($array as $k => $v) {
            if ($v > $max) {
                $max = $v;
                $idx = $k;
            }
        }
        return $idx;
    }

    // Add confusion matrix, precision, recall if needed.
    // For now, accuracy is the most requested.
}
