<?php

class Bias extends ErrorHandler {

    /**
     * @var int
     */
    private $minRandValue = -1000000;

    /**
     * @var int
     */
    private $maxRandValue = 1000000;

    /**
     * Set the Min Value to Rand Bias
     *
     * @param int $minValue
     * @return void
     */
    public function setMinRandBias($minValue) : void {
        $this->isInteger($minValue, __FUNCTION__);
        $this->minRandValue = $minValue;
    }

    /**
     * Set the Max Value to Rand Bias
     *
     * @param int $maxValue
     * @return void
     */
    public function setMaxRandBias($maxValue) : void {
        $this->isInteger($maxValue, __FUNCTION__);
        $this->maxRandValue = $maxValue;
    }

    /**
     * Generate random Bias
     * 
     * ((((float) rand() / (float) getrandmax()) *2) -1);
     *
     * @return float
     */
    public function getRandBias() : float {
        $randValue = (mt_rand($this->minRandValue, $this->maxRandValue) / $this->maxRandValue);
        return $randValue;
    }

}