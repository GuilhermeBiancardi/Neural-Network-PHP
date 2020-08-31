<?php

class Tanh extends ErrorHandler{

    /**
     * Tanh
     *
     * @param float $number
     * @return float
     */
    public function value($number) : float {
        return (2/(1 + exp(-2*$number)) -1);
    }

    /**
     * Derivate of Tanh
     *
     * @param float $sigmoid
     * @return float
     */
    public function differentiate($number) : float {
        return (1- exp(_tanh($number)));
    }

}