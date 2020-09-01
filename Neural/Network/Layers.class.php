<?php

class Layers extends Weight {

    /**
     * @var array
     */
    private $layers = [];

    /**
     * Get the value of layers
     */ 
    public function getLayers() : array {
        return $this->layers;
    }

    /**
     * Set the layers value
     *
     * @param array $layers
     * @return void
     */
    public function setLayers($layers) : void {
        $this->isArray($layers, __FUNCTION__);
        $this->layers = $layers;
    }

    /**
     * Generate Layers
     * 
     * $activation expected to default: array, but receive boll sometimes.
     * $bias expected to default: object, but receive boll sometimes.
     *
     * @param array $layers
     * @param array $activation
     * @param object $bias
     * @return void
     */
    public function generateLayers($layers, $activation = false, $bias = false) : void {
        $nodeIndex = 0;
        foreach(array_values($layers) as $layerKey => $layerValue) {
            $nodeIndex = $this->createLayer($layerKey, $layerValue, $nodeIndex, $activation, $bias);
        }
    }

    /**
     * Create and populate layers
     * 
     * $activation expected to default: array, but receive boll sometimes.
     * $bias expected to default: object, but receive boll sometimes.
     *
     * @param int $layerIndex
     * @param int $sizeLayers
     * @param array $layerValue
     * @param integer $nodeIndex
     * @param array $activation
     * @param object $bias
     * @return int
     */
    private function createLayer($layerIndex, $layerValue, $nodeIndex, $activation, $bias) : int {

        $this->layers[$layerIndex] = [
            "activation" => 0,
            "synapses" => [],
            "value" => [],
            "bias" => [],
        ];

        if($activation) {
            if($layerIndex != 0) {
                if(array_key_exists(($layerIndex -1), $activation)) {
                    $this->layers[$layerIndex]["activation"] = $activation[($layerIndex -1)];
                } else {
                    $this->setError("Activation Function not found for Layer Index:{" . $layerIndex . "}");
                }
            }
        }

        for($index = 0; $index < $layerValue; $index++) {

            if($bias) {
                if($layerIndex != 0) {
                    $this->layers[$layerIndex]["bias"][][] = $bias->getRandBias();
                }
            } else {
                $this->layers[$layerIndex]["bias"][][] = 0;
            }

            $this->layers[$layerIndex]["value"][$nodeIndex][] = $this->setNodeValue(0);
            $nodeIndex++;
        }

        return $nodeIndex;
    }

}