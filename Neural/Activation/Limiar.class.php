<?php

class Limiar extends ErrorHandler {

    /**
     * @var float
     */
    private $limiar;

    /**
     * Limiar
     *
     * @param float $number
     * @return int
     */
    public function value($number) {
        $this->limiar = $number < 0 ? 0 : 1;
        return $this->limiar;
    }

}

?>