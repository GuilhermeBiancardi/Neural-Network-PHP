<?php

class MultiLayerPerceptron extends Synapses {

    /**
     * @var array
     */
    private $inputs;

    /**
     * @var array
     */
    private $populateLayers;

    /**
     * Get the Populate Layers
     *
     * @return array
     */
    public function getPopulateLayers() : array {
        return $this->populateLayers;
    }

    /**
     * Set populate layers
     *
     * @param array $populateLayers
     * @return void
     */
    public function setPopulateLayers($populateLayers) : void {
        $this->isArray($populateLayers);
        $this->populateLayers = $populateLayers;
    }

    /**
     * Get teh Inputs
     *
     * @return array
     */
    public function getInputs() : array {
        return $this->inputs;
    }

    /**
     * Set the Inputs
     *
     * @param array $inputs
     * @return void
     */
    public function setInputs($inputs) : void {
        $this->isArray($inputs, __FUNCTION__);
        $this->inputs = $inputs;
    }

    /**
     * Prepare Structure of Multi Layer Perceptron
     *
     * @param array $layers
     * @param array/bool $activation
     * @param object/bool $bias
     * @return void
     */
    public function prepareMultiLayerStructure($layers, $activation = false, $bias = false) : void {
        $this->isArray($layers, __FUNCTION__);
        $this->amountLayers($layers, __FUNCTION__);
        $activation ? $this->isArray($activation, __FUNCTION__) : "";
        $bias ? $this->isObject($bias, __FUNCTION__) : "";
        $this->generateLayers($layers, $activation, $bias);
        $this->generateSynapses();
        $this->populateLayers = $this->getLayers();
    }

    /**
     * Execute the Feed Forward
     *
     * @return void
     */
    public function feedForward() : void {
        $this->setInputLayerNodesValues();
        $this->regressionLinear();
    }

    /**
     * Set the Nodes Values in Input Layer
     *
     * @return void
     */
    private function setInputLayerNodesValues() : void {
        foreach(array_keys($this->populateLayers[0]["value"]) as $inputLayerKey) {
            if(array_key_exists($inputLayerKey, $this->inputs)) {
                $this->populateLayers[0]["value"][$inputLayerKey][0] = $this->inputs[$inputLayerKey];
            } else {
                $this->populateLayers[0]["value"][$inputLayerKey][0] = 0;
            }
        }
    }

    /**
     * Prepare to execute regression linear
     *
     * @return void
     */
    public function regressionLinear() : void {

        $layers = $this->populateLayers;

        for($indexLayers = 0; $indexLayers < (count($layers) -1); $indexLayers++) {
            $regressionLinear = $this->multiplyNonLinearVector($layers[($indexLayers +1)]["synapses"], $layers[$indexLayers]["value"]);
            $regressionLinear = $this->addValuesToVector($regressionLinear, $layers[($indexLayers +1)]["bias"]);

            $regressionLinear = $this->mapVector($regressionLinear, [$layers[($indexLayers +1)]["activation"]], function($activation, $value) {
                if(is_object($activation[0])) {
                    return $activation[0]->value($value);
                } else {
                    return $value;
                }
            });

            $layers[($indexLayers +1)]["value"] = $regressionLinear;
        }

        $this->populateLayers = $layers;
    }

    /**
     * Multiply values between two vectors non linear
     * 
     * ex:
     * 
     * |0, 0, 0, 0|  X  |0, 0|
     * |0, 0, 0, 0|     |0, 0|
     *
     * @param array $firstVector
     * @param array $secondVector
     * @return array
     */
    public function multiplyNonLinearVector($firstVector, $secondVector) : array {

        $this->checkVectorNonLinearMultiplyCompatibility($firstVector, $secondVector);
        $newVector = $this->newProductVectorNonLinear($firstVector, $secondVector);

        $newVector = $this->mapVector($newVector, [$firstVector, $secondVector], function($vectors, $value, $row, $col) {
            $sum = $value * 0;
            foreach(array_keys($vectors[0][$row]) as $firstVectorKey) {
                $operation = $vectors[0][$row][$firstVectorKey] * $vectors[1][$firstVectorKey][$col];
                $sum += $operation;
            }
            return $sum;
        });

        return $newVector;
    }

    /**
     * Check if vectors of multiply are compatibles to multiply operation
     *
     * @param array $firstVector
     * @param array $secondVector
     * @return void
     */
    public function checkVectorNonLinearMultiplyCompatibility($firstVector, $secondVector) : void {
        if(isset($firstVector[0]) && isset($secondVector[0])) {
            $firstVectorCols = count($firstVector[0]);
            $secondVectorRows = count($secondVector);
            if($firstVectorCols != $secondVectorRows) {
                print_r($firstVector);
                print_r($secondVector);
                echo $firstVectorCols . " - " . $secondVectorRows . PHP_EOL;
                $this->setError("Columns of first vector must match rows of second vector.");
            }
        } else {
            $this->setError("Some vector to be empty.");
        }
    }

    /**
     * Create new empty vector
     *
     * @param array $firstVector
     * @param array $secondVector
     * @return array
     */
    public function newProductVectorNonLinear($firstVector, $secondVector) : array {
        $newVector = [];
        for($indexFirstVector = 0; $indexFirstVector < count($firstVector); $indexFirstVector++) {
            for($indexSecondVector = 0; $indexSecondVector < count($secondVector[0]); $indexSecondVector++) {
                $newVector[$indexFirstVector][$indexSecondVector] = 0;
            }
        }
        return $newVector;
    }

    /**
     * Map the each values on the informed vector
     *
     * @param array $vector
     * @param array $param
     * @param function $executable
     * @return array
     */
    public function mapVector($vector, $param, $executable) : array {
        foreach(array_keys($vector) as $vectorRowKey) {
            foreach(array_keys($vector[$vectorRowKey]) as $vectorColKey) {
                $vectorValue = $vector[$vectorRowKey][$vectorColKey];
                $vector[$vectorRowKey][$vectorColKey] = $executable($param, $vectorValue, $vectorRowKey, $vectorColKey);
            }
        }
        return $vector;
    }

    /**
     * Sum the vectors values
     *
     * @param array $vector
     * @param array $valuesToSum
     * @return array
     */
    public function addValuesToVector($vector, $valuesToSum) : array {
        $this->checkVectorSSCompatibility($vector, $valuesToSum);
        $vector = $this->mapVector($vector, [$valuesToSum], function($toSum, $value, $row, $col) {
            return $toSum[0][$row][$col] + $value;
        });
        return $vector;
    }

    /**
     * Check if vectors of sum are compatibles to sum operation
     *
     * @param array $firstVector
     * @param array $secondVector
     * @return void
     */
    public function checkVectorSSCompatibility($firstVector, $secondVector) : void {
        if(is_array($secondVector)) {
            if(isset($firstVector[0]) && isset($secondVector[0])) {
                $firstVectorRows = count($firstVector);
                $firstVectorCols = count($firstVector[0]);
                $secondVectorRows = count($secondVector);
                $secondVectorCols = count($secondVector[0]);
                if($firstVectorRows != $secondVectorRows) {
                    $this->setError("Rows of first vector must match rows of second vector.");
                }
                if($firstVectorCols != $secondVectorCols) {
                    $this->setError("Columns of first vector must match columns of second vector.");
                }
            } else {
                $this->setError("Some vector to be empty.");
            }
        }
    }

    /**
     * Subtract the vectors values
     *
     * @param array $vector
     * @param array $valuesToSubtract
     * @return array
     */
    public function subtractValuesToVector($vector, $valuesToSubtract) : array {
        $this->checkVectorSSCompatibility($vector, $valuesToSubtract);
        $vector = $this->mapVector($vector, [$valuesToSubtract], function($toSubtract, $value, $row, $col) {
            return $value - $toSubtract[0][$row][$col];
        });
        return $vector;
    }

    /**
     * Multiply values between two vectors non linear
     *
     * @param array $firstVector
     * @param array $secondVector
     * @return array
     */
    public function multiplyLinearVector($firstVector, $secondVector) : array {

        $this->checkVectorSSCompatibility($firstVector, $secondVector);

        $newVector = $this->mapVector($firstVector, [$firstVector, $secondVector], function($vectors, $value, $row, $col) {
            if(is_array($vectors[1])) {
                return ($value * $vectors[1][$row][$col]);
            } else {
                return ($value * $vectors[1]);
            }
        });

        return $newVector;
    }

    /**
     * Transpose vector
     *
     * @param array $vector
     * @return array
     */
    public function transposeVector($vector) : array {
        $newVector = [];
        for($indexColVector = 0; $indexColVector < count($vector[0]); $indexColVector++) {
            for($indexRowVector = 0; $indexRowVector < count($vector); $indexRowVector++) {
                $newVector[$indexColVector][$indexRowVector] = $vector[$indexRowVector][$indexColVector];
            }
        }
        return $newVector;
    }

    /**
     * Get output
     *
     * @return array
     */
    public function getResponse() : array {
        $this->feedForward();
        $output = end($this->populateLayers);
        return $output["value"];
    }

    /**
     * Export data
     *
     * @return string
     */
    public function exportData() : string {
        $dataExport = [ "layers" => $this->getPopulateLayers() ];
        $dataExport = base64_encode(serialize($dataExport));
        return $dataExport;
    }

    /**
     * Import Data
     *
     * @param string $data
     * @return void
     */
    public function importData($data) : void {
        $this->isBase64($data, __FUNCTION__);
        $data = base64_decode($data);
        $this->isSerialize($data, __FUNCTION__);
        $dataImport = unserialize($data);
        $this->setPopulateLayers($dataImport["layers"]);
    }

}

?>