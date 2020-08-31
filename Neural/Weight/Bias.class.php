<?php

class Bias extends ErrorHandler {

    /**
     * @var int
     */
    private $minValue;

    /**
     * @var int
     */
    private $maxValue;
    
    /**
     * @var int
     */
    private $normalize;
    
    /**
     * @var float 
     */
    private $bias;

    public function __construct($minValue = 1, $maxValue = 100) {
        $this->isInteger($minValue, __FUNCTION__);
        $this->isInteger($maxValue, __FUNCTION__);
        $this->minValue = $minValue;
        $this->maxValue = $maxValue;
        $this->normalize = ($maxValue * 10);
    }
    
    /**
     * Get de Rand Bias
     *
     * @return float
     */
    public function getRandBias() : float {
        $randValue = ((((float) rand()/(float) getrandmax()) *2) -1);
        return $randValue;
    }

}