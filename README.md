# Redes Neurais em PHP

## Identificador de Padrões

Este algoritmo é excelente para identificar padrões, podendo ser utilizado para várias finalidades.

## Inclusão da Classe

```php

include_once "diretório/Neural/Structure.class.php";

```

## Chamada da Classe

Backpropagation: Nada de especial na chamada.

```php

$nn = new BackPropagation();

```

Multi Layer Perceptron: Nada de especial na chamada.

```php

$nn = new MultiLayerPerceptron();

```

## Configurações:

A rede pode ser alterada para atender várias necessidades, por isso deve-se configura-la de acordo.

As configurações abaixo podem ser utilizadas em ambas as redes:

```php

// Devem ser instanciadas antes do Prepare Structure

/**
 * 
 * É obrigatório o mínimo de 3 camadas da rede 1 de entrada
 * 1 oculta e 1 de saída, no exemplo abaixo temos:
 * 
 * 3 Camadas.
 * 1º Camada: 2 Neurônios (Camada de entrada)
 * 2º Camada: 4 Neurônios (Camada oculta)
 * 3º Camada: 1 Neurônio (Camada de saída) 
 * 
 */
$structure = [2,4,1];

/**
 * Deve-se informar as funções de ativação que
 * serão usadas nas camadas ocultas e na camada
 * de saída, como temos 1 camada oculta e 1
 * camada de saída serão 2 funções.
 */
$activation = [new Sigmoid(), new Sigmoid()];

/**
 * Criamos um Bias para a rede.
 */
$bias = new Bias();

/**
 * Define o número mínio para geração de bias
 * O padrão é -1000000
 */
$bias->setMinRandBias(-200);

/**
 * Define o número máximo para geração de bias
 * O padrão é 1000000
 */
$bias->setMaxRandBias(200)

/**
 * Define o número mínio para geração de pesos
 * O padrão é -1000000
 */
$nn->setMinRandWeight(-200);

/**
 * Define o número máximo para geração de pesos
 * O padrão é 1000000
 */
$nn->setMaxRandWeight(200);

/**
 * Define se o peso gerado entre o número mínimo e o
 * número máximo ex: 957.843, será normalizado 
 * resultando em 0,957843
 * Padrão é true
 */
$nn->normalizeWeight(false);

// Devem ser instanciadas após o Prepare Structure

/**
 * Pega a resposta da rede.
 */
$nn->getResponse();

```

Configurações exclusivas para BackPropagation:

```php

/**
 * BackPropagation Prepare Structure
 */
$nn->prepareStructure($structure, $activation, $bias);

/**
 * Define um novo Learn Rate.
 * O padrão é 0.1
 */ 
$nn->setLearnRate(0.5);

/**
 * Treina a rede executando o feedForward
 * e backpropagation 1 vez
 */
$nn->train();

```

Configurações exclusivas para Multi Layer Perceptron:

```php

/**
 * Multi Layer Perceptron Prepare Structure
 * 
 * Neste caso $activation e $bias não
 * são obrigatórios para o funcionamento
 * da rede, podendo não serem informados.
 * 
 * Padrão para $activation = false
 * Padrão para $bias = false
 * 
 */
$nn->prepareMultiLayerStructure($structure, $activation, $bias);

// Executa o feedForward da rede
$nn->feedForward();

```

## Treinando a Rede BackPropagation

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

## Exportando o resultado do Treino BackPropagation ou Rede Multi Layer Perceptron

Em alguns casos a rede pode demorar muito para treinar, e ter que treina-la sempre que precisarmos de alguma resposta pode se tornar algo custoso, uma solução é exportar os dados de treino para que possa ser utilizado posteriormente, isso agiliza muito o processo, pois a rede não precisará ser treinada:

```php

$export = $nn->exportData();

```

## Importando dados do Treino BackPropagation ou Rede Multi Layer Perceptron

Com os dados de treino devidamente guardados, podemos inseri-los na rede, assim não precisaremos treinar a rede sempre que precisar-mos testar algum valor.

```php

$export = "DADOS_EXPORTADOS_DA_REDE";
$nn->importData($export);

```

Aproveitem!!!