<?php

class BackPropagation extends MultiLayerPerceptron {

    /**
     * @var array
     */
    private $expectedResponse;

    /**
     * @var float
     */
    private $learnRate = 0.1;

    /**
     * @var array
     */
    private $outputError;

    /**
     * Get the expected response
     *
     * @return array
     */
    public function getExpectedResponse() : array {
        return $this->expectedResponse;
    }

    /**
     * Set the expected response
     *
     * @param array $expectedResponse
     * @return void
     */
    public function setExpectedResponse($expectedResponse) : void {
        $this->isArray($expectedResponse);
        if(isset($expectedResponse[0][0])) {
            $this->expectedResponse = $expectedResponse;
        } else {
            $this->expectedResponse = $this->convertArrayToVector($expectedResponse);
        }
    }

    /**
     * Get learn rate
     *
     * @return float
     */
    public function getLearnRate() : float {
        return $this->learnRate;
    }

    /**
     * Set learn rate
     *
     * @param float $learnRate
     * @return void
     */
    public function setLearnRate($learnRate) : void {
        $this->isFloat($learnRate);
        $this->learnRate = $learnRate;
    }

    /**
     * Get output error
     *
     * @return array
     */
    public function getOutputError() : array {
        return $this->outputError;
    }

    /**
     * Set output error
     *
     * @return void
     */
    public function setOutputError($outputError) : void {
        $this->isArray($outputError);
        $this->outputError = $outputError;
    }

    /**
     * Prepare Structure of Multi Layer Perceptron
     *
     * @param array $layers
     * @param array $activation
     * @param object $bias
     * @return void
     */
    public function prepareStructure($layers, $activation, $bias) : void {
        $this->isArray($layers, __FUNCTION__);
        $this->amountLayers($layers, __FUNCTION__);
        $this->isArray($activation, __FUNCTION__);
        $this->isObject($bias, __FUNCTION__);
        $this->generateLayers($layers, $activation, $bias);
        $this->generateSynapses();
        $this->setPopulateLayers($this->getLayers());
    }

    /**
     * Train dataset
     *
     * @return void
     */
    public function train() {
        $this->feedForward();
        $this->calculateOutputError();
        $this->backPropagate();
    }

    /**
     * Convert array to vector
     *
     * @param array $array
     * @return array
     */
    private function convertArrayToVector($array) : array {
        $vector = [];
        foreach($array as $arrayKey => $arrayValue) {
            $vector[$arrayKey][] = $arrayValue;
        }
        return $vector;
    }

    /**
     * Calculate the output error
     *
     * @return void
     */
    private function calculateOutputError() : void {
        $layers = $this->getPopulateLayers();
        $output = end($layers);
        $this->outputError = $this->subtractValuesToVector($this->expectedResponse, $output["value"]);
    }

    /**
     * Back propagate the error and adjusts weight and bias
     *
     * @return void
     */
    private function backPropagate() {
        $layers = $this->getPopulateLayers();
        $gradient = $this->calculateGradientsOutputLayer($layers, array_key_last($layers));
        
        for($indexLayers = (count($layers) -1); $indexLayers > 0; $indexLayers--) {
            $transpose = $this->transposeVector($layers[($indexLayers -1)]["value"]);
            $deltaWeight = $this->multiplyNonLinearVector($gradient, $transpose);
            $weightAdjusts = $this->addValuesToVector($layers[$indexLayers]["synapses"], $deltaWeight);
            $layers[$indexLayers]["synapses"] = $weightAdjusts;
            $biasAdjusts = $this->addValuesToVector($layers[$indexLayers]["bias"], $gradient);
            $layers[$indexLayers]["bias"] = $biasAdjusts;
            $transposeWeight = $this->transposeVector($weightAdjusts);
            $errorAdjusts = $this->multiplyNonLinearVector($transposeWeight, $this->outputError);
            $this->outputError = $errorAdjusts;
            $gradient = $this->calculateGradientsOutputLayer($layers, ($indexLayers -1));
        }

        $this->setPopulateLayers($layers);
    }

    /**
     * Calculate gradient output layer
     *
     * @param array $layers
     * @return array
     */
    private function calculateGradientsOutputLayer($layers, $indexLayer) : array {
        $gradient = $this->mapVector($layers[$indexLayer]["value"], [$layers[$indexLayer]["activation"]], function($derivation, $value) {
            if(is_object($derivation[0])) {
                return $derivation[0]->differentiate($value);
            } else {
                return $value;
            }
        });
        $gradient = $this->multiplyLinearVector($gradient, $this->outputError);
        $gradient = $this->multiplyLinearVector($gradient, $this->learnRate);
        return $gradient;
    }

}

?>