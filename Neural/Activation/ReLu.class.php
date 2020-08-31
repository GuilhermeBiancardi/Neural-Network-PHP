<?php

class ReLu extends ErrorHandler {

    /**
     * @var float
     */
    private $relu;

    /**
     * @var float
     */
    private $differentiate;

    /**
     * ReLu
     *
     * @param float $number
     * @return float
     */
    public function value($number) : float {
        $this->isNumeric($number, __FUNCTION__);
        $this->relu = max(0, $number);
        return $this->relu;
    }

    /**
     * Derivate of ReLu
     *
     * @param float $number
     * @return int
     */
    public function differentiate($number) : int {
        $this->isNumeric($number, __FUNCTION__);
        $number >= 0 ? $this->differentiate = 1 : $this->differentiate = 0;
        return $this->differentiate;
    }

}

?>