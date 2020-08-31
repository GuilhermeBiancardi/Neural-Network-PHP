# Redes Neurais em PHP

## Identificador de Padrões

Este algoritmo é excelente para identificar padrões, podendo ser utilizado para várias finalidades.

## Inclusão da Classe

```php
    include_once "diretório/Neural/Structure.class.php";
```

## Chamada da Classe

Nada de especial na chamada:

```php
    $nn = new BackPropagation();
```

## Configurações:

A rede pode ser alterada para atender várias necessidades, por isso deve-se configura-la de acordo:

```php
    /**
     * 2 Neurônios na camada de entrada
     * 4 Neurônios na camada oculta
     * 1 na camada de saída 
     */
    $structure = [2,4,1];
    $activation = [new Sigmoid(), new Sigmoid(), new Sigmoid()];
    $bias = new Bias();
    $nn->prepareStructure($structure, $activation, $bias);

    // Define um novo Learn Rate
    $nn->setLearnRate(0.5);
```

## Treinando a Rede

Após definir todos os valores basta chamar a função de treinamento, se todos os valores estiverem corretos a rede irá treinar até chegar na resposta esperada, ex:

```php

    $nn = new BackPropagation();

    $structure = [2,4,1];
    $activation = [new Sigmoid(), new Sigmoid(), new Sigmoid()];
    $bias = new Bias();
    $nn->prepareStructure($structure, $activation, $bias);

    // XOR problem
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

```

## Exportando o treino

Em alguns casos a rede pode demorar muito para treinar, e ter que treina-la sempre que precisarmos de alguma resposta pode se tornar algo custoso, uma solução é exportar os dados de treino para que possa ser utilizado posteriormente, isso agiliza muito o processo, pois a rede não precisará ser treinada:

```php
    $export = $nn->exportData();
```

## Importando dados de treinos

Com os dados de treino devidamente guardados, podemos inseri-los na rede, assim não precisaremos treinar a rede sempre que precisar-mos testar algum valor.

```php
    $export = "DADOS_EXPORTADOS_DA_REDE";
    $nn->importData($export);
```

Aproveitem!!!
