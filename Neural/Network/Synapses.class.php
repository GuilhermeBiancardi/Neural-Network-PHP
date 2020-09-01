<?php

class Synapses extends Layers {

    /**
     * @var array
     */
    private $synapses = [];

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