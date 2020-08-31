<?php

class Synapses extends Layers {

    /**
     * @var array
     */
    private $synapses = [];

    /**
     * @var int
     */
    private $minRandValue = -100000000;

    /**
     * @var int
     */
    private $maxRandValue = 1000000000;

    /**
     * Get the synapses
     *
     * @return array
     */
    public function getSynapses() : array {
        return $this->synapses;
    }

    /**
     * Set the synapses
     *
     * @return void
     */
    public function setSynapses($synapses) : void {
        $this->isArray($synapses, __FUNCTION__);
        $this->synapses = $synapses;
    }

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
     * Generate random weight
     *
     * @return float
     */
    private function getWeight() : float {
        $randValue = ((((float) rand()/(float) getrandmax()) *2) -1);
        return $randValue;
    }

    /**
     * Create Synapses
     *
     * @return void
     */
    public function generateSynapses() : void {
        $layers = $this->getLayers();
        foreach(array_keys($layers) as $indexLayer) {
            $this->generateLigation($indexLayer);
        }
    }

    /**
     * Create Synapses of Nodes
     *
     * @param int $layerIndex
     * @param array $layers
     * @return void
     */
    private function generateLigation($layerIndex) : void {

        $layers = $this->getLayers();
        if(array_key_exists(($layerIndex +1), $layers)) {
            $thisLayerNodes = $layers[$layerIndex]["value"];
            $nextLayerNodes = $layers[($layerIndex +1)]["value"];
            for($indexNextLayerNodes = 0; $indexNextLayerNodes < count($nextLayerNodes); $indexNextLayerNodes++) {
                for($indexThisLayerNodes = 0; $indexThisLayerNodes < count($thisLayerNodes); $indexThisLayerNodes++) {
                    $layers[($layerIndex +1)]["synapses"][$indexNextLayerNodes][$indexThisLayerNodes] = $this->getWeight();
                }
            }
            $this->setLayers($layers);
        }

    }

}