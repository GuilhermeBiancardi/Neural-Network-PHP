<?php

class Sigmoid extends ErrorHandler{

    /**
     * @var float
     */
    private $sigmoid;

    /**
     * @var float
     */
    private $differentiate;

    /**
     * Sigmoid
     *
     * @param float $number
     * @return float
     */
    public function value($number) : float {
        $this->isNumeric($number, __FUNCTION__);
        $this->sigmoid = 1 / (1 + exp(-1.0 * $number));
        return $this->sigmoid;
    }

    /**
     * Derivate of Sigmoid
     *
     * @param float $sigmoid
     * @return float
     */
    public function differentiate($sigmoid) : float {
        $this->isNumeric($sigmoid, __FUNCTION__);
        $this->differentiate = ($sigmoid * (1 - $sigmoid));
        return $this->differentiate;
    }

}