<?php

class Node extends ErrorHandler {

    /**
     * @var float
     */
    private $node;

    /**
     * @param float $value
     * @return float
     */
    public function setNodeValue($value) : float {
        $this->isNumeric($value, __FUNCTION__);
        $this->node = $value;
        return $this->node;
    }

}