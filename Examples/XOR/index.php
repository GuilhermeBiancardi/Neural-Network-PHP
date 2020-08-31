<?php

include_once "../../Neural/Structure.php";

$nn = new BackPropagation();

$xor = [
    1 => [
        "inputs" => [0, 0],
        "output" => [0]
    ],
    2 => [
        "inputs" => [0, 1],
        "output" => [1]
    ],
    3 => [
        "inputs" => [1, 0],
        "output" => [1]
    ],
    4 => [
        "inputs" => [1, 1],
        "output" => [0]
    ]
];

$nn->prepareStructure([2,4,1], [new Sigmoid(), new Sigmoid()], new Bias());
$nn->setLearnRate(0.5);

for($loop = 0; $loop < 1000; $loop++) {
    for($i = 1; $i <= 4; $i++) {
        $nn->setInputs($xor[$i]["inputs"]);
        $nn->setExpectedResponse($xor[$i]["output"]);
        $nn->train();
    }
}

$nn->setInputs($xor[1]["inputs"]);
print_r($nn->getResponse());

$nn->setInputs($xor[2]["inputs"]);
print_r($nn->getResponse());

$nn->setInputs($xor[3]["inputs"]);
print_r($nn->getResponse());

$nn->setInputs($xor[4]["inputs"]);
print_r($nn->getResponse());

?>