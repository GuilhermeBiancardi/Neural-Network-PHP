# Redes Neurais em PHP

## Identificador de Padrões

Este algoritmo é excelente para identificar padrões, podendo ser utilizado para várias finalidades.

## Inclusão da Classe

```php
    include_once "diretorio/NeuralNetwork.class.php";
```

## Chamada da Classe

Nada de especial na chamada:

```php
    $nn = new NeuralNetwork();
```

## Configurações:

A rede pode ser alterada para atender várias necessidades, por isso deve-se configura-la de acordo:

```php
    $configuration = Array(
        // Parâmetros
    );
    $nn->setConfiguration($configuration);
```

## Parâmetros de Configuração:

Nosso **Array** de configurações necessita dos seguintes parâmetros (falaremos de cada um separadamente):

```php
    $configuration = Array(
        // Required: Estrutura da Nossa Rede
        "structure" => Array(2,2,2),
        
        // Required: Entradas
        "values" => Array(0, 0),

        // Required: Saída Desejada
        "response" => Array(0, 0),

        // Optional: Taxa de Aprendizado
        "error_learn" => 0.01,

        // Optional: Taxa de aceitação de saida
        "error" => 0.0001,

        // Optional: Variação dos Pesos
        "weight_variation" => 100,

        // Optional: Variação dos Bias
        "bias_variation" => 100,

        // Optional: Tamanho da memória alocada pelo PHP para execução
        "memory_limit" => "512M"
    );
```

**structure:** Define o tamanho da nossa rede, sendo obrigatório ter no mínimo 3 camadas, uma de entrada, uma oculta e uma de saída, uma rede 2,2,2 por exemplo terá 2 valores na camada de entrada, 2 neurônios na camada oculta e 2 valores de saída. Já uma rede 2,3,3,1 terá 2 valores de entrada, 2 camadas ocultas com 3 neurônios cada e 2 saídas.

**values:** Declaramos aqui nossas entradas em um Array.

**response:** Temos também que informar nossa saída esperada, para que nossa rede aprenda com uma base teste antes de ser colocada a prova.

**error_learn:** Define a taxa de aprendizado da nossa rede, quanto menor for o número mais preciso será o resultado (Diminuir o número pode custar mais recursos computacionais como memória RAM e irá demorar mais para a rede ser treinada). Esse campo não é Obrigatório e seu valor padrão é 0.01

**error:** Podemos informar uma taxa de aceitação para que a rede pare ao chegar a um erro aceitável, a rede nunca irá chegar no valor esperado e sim se aproximar cada vez mais dele, esse valor determina a quantidade aproximada que a rede deve aceitar para parar a execução. Esse campo não é Obrigatório e seu valor padrão é 0.0001

**weight_variation:** Número que determinará a variação de pesos que serão gerados para as sinapses, ele deve estar entre +1 e +X. Esse campo não é Obrigatório e seu valor padrão é 100.

**bias_variation:** Número que determinará a variação de pesos que serão gerados para os bias, ele deve estar entre +1 e +X. Esse campo não é Obrigatório e seu valor padrão é 100.

**memory_limit:** Quantidade de memória que deve ser alocada pelo PHP para execução da rede. Quanto mais recurso for disponibilizado para a rede, mais complexa a rede poderá ser. Esse campo não é Obrigatório e seu valor padrão é o padrão no php.ini.

## Inserindo os pesos manualmente

IMPORTANTE!!! A rede gera os pesos aleatóriamente, mas é possível importar pesos pré-definidos manualmente, caso eles não forem informados a rede irá gerar pesos aleatórios.

**Sinapses:** Caso precise testar a rede com pesos já existentes é possível inserir os mesmos na rede, mas para isso é preciso entender a lógica organizacional da rede:

Exemplo de Rede 2x2x1

Os 2 Perceptrons de entrada receberão o nome de 1 e 2. Os 2 Perceptrons da Camada Oculta receberão o nome de 3 e 4 O Perceptron de Saída receberá o nome de 5

Obs: se existirem mais valores o número será sequencial.

A rede ficaria assim:

```
 1 --- 3
   \ /   \
    X      5
   / \   /
 2 --- 4
 ```

Então a Sinapse que liga o Perceptron
1 ao Perceptron 3 recebera o nome w1-3
logo temos que informar os pesos da
seguinte forma:

```php
    $pesos = Array(

        /*
         * Entrada 1 propagando seu valor para os
         * perceptrons da camada oculta
         */ 
        "w1-3" => 0.15,
        "w1-4" => 0.25,

        /*
         * Entrada 2 propagando seu valor para os
         * perceptrons da camada oculta
         */
        "w2-3" => 0.2,
        "w2-4" => 0.3,

        /*
         * Perceptron 1 (com nome 3) propagando seu valor para os
         * perceptrons de saída
         */
        "w3-5" => 0.4,

        /*
         * Perceptron 2 (com nome 4) propagando seu valor para os
         * perceptrons de saída
         */
        "w4-5" => 0.4,

    );
```

Esse array deve ser passado a rede "serialized":

```php
    $nn->setWeight(serialize($pesos));
```

**Bias:** Com os bias a lógica é parecida:

Seguindo o exemplo anterior da nossa Rede (2x2x1). Cada coluna representa uma camada da Rede, onde a camada de entrada será 0 e a próxima será 1 e assim sucessivamente até a camada de saída.

Obs: a camada 0 não deve receber um Bias!

```
   0         1     2
  ___       ___
 | 1 | --- | 3 |
 |   | \ / |   |\ ___
 |   |  X  |   | | 5 |
 |   | / \ |   |/ ---
 | 2 | --- | 4 |
  ---       ---
```

Então deve-se informar os Bias da seguinte forma:

```php
    $bias = Array(
        "b1" => 0.5,
        "b2" => 0.2,
    );
```

Esse array deve ser passado a rede "serialized".

```php
    $nn->setBias(serialize($bias));
```

## Treinando a Rede

Após definir todos os valores basta chamar a função de treinamento, se todos os valores estiverem corretos a rede irá treinar até chegar na resposta esperada, ex:

```php

    $nn = new NeuralNetwork();

    $configuration = Array(

        "structure" => Array(2,2,1),
        "values" => Array(0.464, 0.482),
        "response" => Array(0.486),
        "memory_limit" => "512M"

    );

    $nn->setConfiguration($configuration);

    $pesos = Array(

        "w1-3" => 0.15,
        "w1-4" => 0.25,

        "w2-3" => 0.2,
        "w2-4" => 0.3,
        
        "w3-5" => 0.4,
        
        "w4-5" => 0.4,

    );


    $bias = Array(
        "b1" => 0.5,
        "b2" => 0.2,
    );

    $nn->setWeight(serialize($pesos));
    $nn->setBias(serialize($bias));

    // Treina a Rede
    $nn->train();

```

## Exibindo a resposta da rede para novos valores

Após a rede estar treinada é possível inserir novos valores para que ela detecte o padrão baseado no seu treino e retorne esses valores:

```php
    
    // Processo de treino...

    // Valores a serem testados
    $valores = Array($x, $y, $z, etc...);
    // Imputamos ele na rede.
    $nn->setValues($valores);

    // Retorno da rede com a melhor resposta baseada nos treinos.
    $saida = $nn->answerBinary();
    print_r($saida);
```

## Exportando o treino

Em alguns casos a rede pode demorar muito para treinar, e ter que treina-la sempre que precisarmos de alguma resposta pode se tornar algo custoso, uma solução é exportar os dados de treino para que possa ser utilizado posteriormente, isso agiliza muito o processo, pois a rede não precisará ser treinada:

```php

    // Processo de treino...

    /**
     * Existem 3 opções predefinidas para exportação:
     * $nn->exportData(ESTRUTURA, DATABASE_INTERNA. EPOCH);
     * todas definidas com TRUE, caso não queira exportar
     * alguns desses dados basta colocar FALSE, ou deixar
     * vazio para retornar todos.
     */
    $export = $nn->exportData();
```

## Importando dados de treinos

Com os dados de treino devidamente guardados, podemos inseri-los na rede, assim não precisaremos treinar a rede sempre que precisar-mos testar algum valor.

```php
    $export = "DADOS_EXPORTADOS_DA_REDE";

    $nn = new NeuralNetwork();
    $nn->importData($export);

    // Valores a serem testados
    $valores = Array($x, $y, $z, etc...);
    // Imputamos ele na rede.
    $nn->setValues($valores);

    // Retorno da rede com a melhor resposta baseada nos treinos.
    $saida = $nn->answerBinary();
    print_r($saida);

```

## Redes Grandes

Sabemos que uma rede com milhões de dados fica impossível a armazenagem dos dados na memória interna da rede, por isso é recomendado que esses dados estejam em uma database. É possível coletar esses dados para armazenagem externa.

```php
    // Dados da rede
    $rede = Array (
        "structure" => Array(3, 3, 2),
        "error" => 0.0001,
        "error_learn" => 0.01,
        // "memory_limit" => "4096M"
    );

    // Adiciono os dados da rede.
    $nn->setConfiguration($rede);

    // Valores de entrada
    $values = Array(
        Array(0.12, 0.13, 0.14),
        Array(0.15, 0.16, 0.17),
        Array(0.18, 0.19, 0.20),
    );

    // Respostas
    $result = Array(
        Array(0.11, 0.12),
        Array(0.13, 0.14),
        Array(0.15, 0.16),
    );

    // Inicia-se o processo de input dos dados na rede para treino
    foreach($values as $key => $value) {

        // Informo os valores
        $nn->setValues($value);

        // Informo as respostas correspondentes
        $nn->setResponse($result[$key]);
        
        // Inicio o treino
        $nn->train();

        // Rede treinada, exporto apenas a estrutura da rede
        $structure = $nn->exportData(true, false, false);

        // Exporto apenas os dados do treino
        $response = $nn->exportData(false, true, false);

        // AQUI VAI O PROCESSO DE GRAVAÇÃO DOS DADOS NO BD

        // Ao importar os dados de estrutura para a rede ela será zerada.
        $nn->importData($structure);

    }

```

## Enviando os dados armazenados no BD

Com os dados armazenados em um banco de dados, é necessário passa-los para a rede quando for preciso encontrar um novo valor nos dados de treino, com a base de dados interna, basta apenas utilizar o "answerBinary()" que a rede já traz o melhor resultado, mas com uma base de dados externa a
comparação desses dados deve ser manual.

```php
    
    // PROCESSO DE CAPTURA DOS DADOS NO BANCO

    /**
     * Assumindo que os dados do BD venham desta forma:
     * Array(
     *     0 => "DADOS EXPORTADOS DA REDE pelo: $nn->exportData(true, false, false);",
     * );
     */
    $structure = "DADOS DE ESTRUTURA VIDO DO BD";

    /**
     * Assumindo que os dados do BD venham desta forma:
     * Array(
     *     0 => "DADOS EXPORTADOS DA REDE pelo: $nn->exportData(false, true, false)",
     *     1 => "DADOS EXPORTADOS DA REDE pelo: $nn->exportData(false, true, false)",
     *     2 => "DADOS EXPORTADOS DA REDE pelo: $nn->exportData(false, true, false)",
     *     etc...
     * );
     */
    $dados = "DADOS DE TREINO VINDO DO BD";

    // Informo a estrutura da rede
    $nn->importData(unserialize($structure[0]));

    $resposta_mais_proxima = 1;
    $resultado = Array();

    // Dados para classificar
    $data = Array(0.1, 0.2, 0.3);

    foreach($dados as $key => $value) {

        // Informo os valores de verificação
        $nn->setValues($data);

        // Pego a resposta da rede no treino informado
        $resposta = $nn->getResponse($value);

        if($resposta["total"] < $resposta_mais_proxima) {
            $resposta_mais_proxima = $resposta["total"];
            $resultado = $resposta;
        }
    }

    // Resposta da rede
    $resposta

```

## Debug dos dados

Mostra algumas informações contidas na rede:

```php
    $nn->debug();
```

Aproveitem!!!
