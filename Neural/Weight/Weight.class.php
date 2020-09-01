<?php

class Weight extends Node {
    
    /**
     * @var int
     */
    private $minRandValue = -1000000;

    /**
     * @var int
     */
    private $maxRandValue = 1000000;

    /**
     * @var boolean
     */
    private $normalize = true;

    /**
     * Set the Min Value to Rand Weight
     *
     * @param int $minValue
     * @return void
     */
    public function setMinRandWeight($minValue) : void {
        $this->isInteger($minValue, __FUNCTION__);
        $this->minRandValue = $minValue;
    }

    /**
     * Set the Max Value to Rand Weight
     *
     * @param int $maxValue
     * @return void
     */
    public function setMaxRandWeight($maxValue) : void {
        $this->isInteger($maxValue, __FUNCTION__);
        $this->maxRandValue = $maxValue;
    }

    /**
     * Normalize Weight
     *
     * @return void
     */
    public function normalizeWeight($normalize) : void {
        $this->isBool($normalize);
        $this->normalize = $normalize;
    }

    /**
     * Generate random weight
     * 
     * ((((float) rand() / (float) getrandmax()) *2) -1);
     *
     * @return float
     */
    public function getWeight() : float {
        $randValue = 1;
        if($this->normalize) {
            $randValue = (mt_rand($this->minRandValue, $this->maxRandValue) / $this->maxRandValue);
        } else {
            $randValue = mt_rand($this->minRandValue, $this->maxRandValue);
        }
        return $randValue;
    }

}

?>