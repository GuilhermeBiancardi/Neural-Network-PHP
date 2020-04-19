<?php

class ImageData {

    private $x = 0;
    private $y = 0;
    private $source = "";
    private $info = Array();

    public function setImage($path) {
        $this->source = imagecreatefrompng($path);
        $this->reset();
        $this->getWidth();
        $this->getHeight();
    }

    private function reset() {
        $this->x = 0;
        $this->y = 0;
        $this->info = Array();
    }

    private function getWidth() {
        $this->x = imagesx($this->source);
    }

    private function getHeight() {
        $this->y = imagesy($this->source);
    }

    private function getColor($x, $y) {
        return imagecolorat($this->source, $x, $y);
    }

    public function getImageInfo() {
        for($i = 0; $i < $this->x; $i++) {
            for($j = 0; $j < $this->y; $j++) {
                $this->info[] = ($this->getColor($i, $j) / 10000000000);
            }
        }
        return $this->info;
    }

}

?>