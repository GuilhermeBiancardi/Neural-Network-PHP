<?php

class ErrorHandler {

    private $typeError = "Alert: ";
    private $dieAble = false;

    /**
     * Set Error Type
     *
     * @param int $type
     * @return void
     */
    private function setTypeError($type = 2) : void {
        switch ($type) {
            case 2:
                $this->typeError = "Warning: ";
                $this->dieAble = true;
            break;
            
            case 1:
                $this->typeError = "Dangerous: ";
                $this->dieAble = true;
            break;
            
            default:
                $this->typeError = "Alert: ";
                $this->dieAble = false;
                break;
        }
    }

    /**
     * Print error message
     *
     * @param string $error
     * @param int $typeError
     * @return void
     */
    public function setError($error, $typeError = 2) : void {
        $this->setTypeError($typeError);
        echo $this->typeError . $error;
        if($this->dieAble) {
            die();
        }
    }

    /**
     * Convert Array to String if necessary
     *
     * @param array $param
     * @return string
     */
    private function paramIsArray($param) : string {
        $paramString = $param;
        if(is_array($paramString)) {
            $paramString = "[" . implode(", ", $paramString) . "]";
        }
        return $paramString;
    }
    
    /**
     * Verify if param is a integer value
     *
     * @param int $value
     * @param string $functionName
     * @return void
     */
    public function isInteger($value, $functionName = "Undefined") : void {
        if(!is_integer($value)) {
            $this->setError("The function:{" . $functionName . "}:param:{'" . gettype($value) . ":" . $this->paramIsArray($value) . "'} is not a integer value.");
        }
    }
    
    /**
     * Verify if param is a float value
     *
     * @param int $value
     * @param string $functionName
     * @return void
     */
    public function isFloat($value, $functionName = "Undefined") : void {
        if(!is_float($value)) {
            $this->setError("The function:{" . $functionName . "}:param:{'" . gettype($value) . ":" . $this->paramIsArray($value) . "'} is not a float value.");
        }
    }

    /**
     * Verify if param is a numeric value
     *
     * @param number $value
     * @param string $functionName
     * @return void
     */
    public function isNumeric($value, $functionName = "Undefined") : void {
        if(!is_numeric($value)) {
            $this->setError("The function:{" . $functionName . "}:param:{'" . gettype($value) . ":" . $this->paramIsArray($value) . "'} is not a numeric value.");
        }
    }

    /**
     * Verify if param is a array value
     *
     * @param array $value
     * @param string $functionName
     * @return void
     */
    public function isArray($value, $functionName = "Undefined") : void {
        if(!is_array($value)) {
            $this->setError("The function:{" . $functionName . "}:param:{'" . gettype($value) . ":" . $this->paramIsArray($value) . "'} is not a array value.");
        }
    }

    /**
     * Verify if param is a object value
     *
     * @param object $value
     * @param string $functionName
     * @return void
     */
    public function isObject($value, $functionName = "Undefined") : void {
        if(!is_object($value)) {
            $this->setError("The function:{" . $functionName . "}:param:{'" . gettype($value) . ":" . $this->paramIsArray($value) . "'} is not a object value.");
        }
    }

    /**
     * Verify if param is a serialize value
     *
     * @param string $value
     * @param string $functionName
     * @return void
     */
    public function isSerialize($value, $functionName = "Undefined") : void {
        if(!@unserialize($value)) {
            $this->setError("The function:{" . $functionName . "}:param:{'" . gettype($value) . ":" . $this->paramIsArray($value) . "'} is not a serializable value.");
        }
    }

    /**
     * Verify if param is a base64 value
     *
     * @param string $value
     * @param string $functionName
     * @return void
     */
    public function isBase64($value, $functionName = "Undefined") : void {
        if(!@base64_decode($value)) {
            $this->setError("The function:{" . $functionName . "}:param:{'" . gettype($value) . ":" . $this->paramIsArray($value) . "'} is not a base64 value.");
        }
    }

    /**
     * Check amount of the layers
     *
     * @param array $layers
     * @return void
     */
    public function amountLayers($layers) : void {
        if(count($layers) < 3) {
            $this->setError("The standard number of layers is an input layer, a hidden layer and an output layer, only amountLayers:{'" . count($layers) . "'} have been informed.", 1);
        }
    }

}

?>