<?php

class Sigmoid extends ErrorHandler{

    /**
     * Linear
     *
     * @param float $number
     * @return float
     */
    public function value($number) {
        return $number;
    }

    /**
     * Derivate of Linear
     *
     * @param float $sigmoid
     * @return float
     */
    public function differentiate($number) {
        return $number;
    }

}