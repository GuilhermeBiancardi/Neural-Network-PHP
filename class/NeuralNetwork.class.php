<?php

/**
 * Rede neural supervisionada
 */

class NeuralNetwork {

    /**
     * Armazena todos os valores da Rede para normalização.
     *
     * @var Array
     */
    private $normalize;
    
    /**
     * Armazena a estrutura da Rede.
     *
     * @var Array
     */
    private $estrutura;
    
    /**
     * Armazena as entradas da Rede.
     *
     * @var Array
     */
    private $entradas;

    /**
     * Armazena as saídas esperadas.
     *
     * @var Array
     */
    private $expected_returns;

    /**
     * Taxa de Aprendizado da Rede.
     *
     * @var Float
     */
    private $taxa_aprendizagem = 0.01;

    /**
     * Armazena o número que definirá o calculo
     * da variante do Peso de casa sinapse,
     * é aconselhável que esse número seja múltiplo
     * de 10. sendo 10 equivalente a pesos de 
     * 0.1 a 0.9, 100 equivalente a 0.01 a 0.99 e
     * assim por diante.
     *
     * @var Integer
     */
    private $variation_weight = 100;

    /**
     * Armazena o número que definirá o calculo
     * da variante do Bias de casa camada,
     * é aconselhável que esse número seja múltiplo
     * de 10. sendo 10 equivalente a pesos de 
     * 0.1 a 0.9, 100 equivalente a 0.01 a 0.99 e
     * assim por diante.
     *
     * @var Integer
     */
    private $variation_bias = 100;

    /**
     * Armazena o número que define quando a
     * Rede deve parar, ou seja, estamos dizendo
     * a Rede quando o erro retornado é aceitável
     * para nosso propósito.
     *
     * @var Float
     */
    private $stop_error = 0.001;

    /**
     * Armazena todos os Perceptrons da Rede.
     *
     * @var Array
     */
    private $perceptrons;

    /**
     * Armazena todos os Pesos das Sinapses da
     * Rede.
     *
     * @var Array
     */
    private $pesos;

    /**
     * Armazena os novos Pesos das Sinapses da
     * Rede após efetuado o Backpropagation.
     *
     * @var Array
     */
    private $novos_pesos;

    /**
     * Armazena os Bias de cada Camada da Rede.
     *
     * @var Array
     */
    private $bias;

    /**
     * Armazena todas as saídas de cada Perceptron
     * após passar pela função de Ativação.
     *
     * @var [type]
     */
    private $activation;

    /**
     * Armazena o Erro Total da Rede após o 
     * FedForward.
     *
     * @var Array
     */
    private $error;

    /**
     * Armazena o Gradient Descent de cada Perceptron
     * efetuados no Backpropagation
     *
     * @var Array
     */
    private $gradient;

    /**
     * Armazena a quantidade de Épocas (Vezes em que a Rede
     * executou o FeedForward e Backpropagation).
     *
     * @var Integer
     */
    private $epoch = 0;

    /**
     * Espaço alocado na memória.
     *
     * @var String
     */
    private $memory_limit = "";

    /**
     * Dados de configuração da rede.
     *
     * @var Array
     */
    private $rede;

    /**
     * Armazena todos os dados de treino da rede.
     *
     * @var Array
     */
    private $train_database = Array();

    public function __construct() {
        //
    }

    /**
     * Incorpora a configuração da rede
     *
     * @param Array $rede
     */
    public function setConfiguration($rede) {
        $this->rede = $rede;
        
        if(isset($rede["memory_limit"])) {
            $this->normalizeMemory($rede["memory_limit"]);
        } else {
            $this->normalizeMemory($this->memory_limit);
        }

        if(isset($rede["structure"])) {
            $this->normalizeEstrutura($rede["structure"]);
        }

        if(isset($rede["error_learn"])) {
            $this->normalizeError($rede["error_learn"]);
        } else {
            $this->normalizeError($this->taxa_aprendizagem);
        }

        if(isset($rede["weight_variation"])) {
            $this->normalizeWeight($rede["weight_variation"]);
        } else {
            $this->normalizeWeight($this->variation_weight);
        }

        if(isset($rede["bias_variation"])) {
            $this->normalizeBias($rede["bias_variation"]);
        } else {
            $this->normalizeBias($this->variation_bias);
        }

        if(isset($rede["error"])) {
            $this->normalizeErrorStop($rede["error"]);
        } else {
            $this->normalizeErrorStop($this->error);
        }
    }

    /**
     * Normaliza os Dados
     *
     * @param Array $rede
     * @return Boolean
     */
    private function normalizeMemory($memory) {

        // Verifica se um valor foi informado.
        if(isset($memory) && $memory != "") {

            $this->memory_limit = $memory;

            // Reserva o espaço informado na Memória
            ini_set("memory_limit", $this->memory_limit);
        }

        // Informo ao normalizador se a informação está OK
        $this->normalize = Array (
            "memory_limit" => true
        );
    }

    /**
     * Normaliza os Dados
     *
     * @param Array $rede
     * @return Boolean/String
     */
    private function normalizeEstrutura($structure) {

        $return = true;

        // Verifica se um valor foi informado.
        if(isset($structure)) {

            // Checa se o valor de structure é um Array
            if(is_array($structure)) {
                
                // Armazena a Estrutura na Rede
                $this->estrutura = $structure;

            } else {
                $return = "Os dados informados não estão armazenados em um Array.";
            }

        } else {
            $return = "Nenhum dado Informado.";
        }

        // Informo ao normalizador se a informação está OK
        $this->normalize = Array (
            "structure" => $return
        );

    }

    /**
     * Normaliza os Dados
     *
     * @param Array $rede
     * @return Boolean/String
     */
    private function normalizeValues($values) {

        $return = true;

        // Verifica se um valor foi informado.
        if(isset($values)) {

            // Checa se o valor de values é um Array
            if(is_array($values)) {

                // Armazena a quantidade de Entradas Informadas.
                $quantidade_entradas_informadas = count($values);

                if($quantidade_entradas_informadas == $this->estrutura[0]) {

                    // Armazena os valores de Entrada na Rede
                    $this->entradas = $values;

                } else {
                    $return = "A quantidade de Neurônios de Entrada não são iguais a quantidade dos Valores de Entrada.";
                }

            } else {
                $return = "Os dados informados não estão armazenados em um Array.";
            }

        } else {
            $return = "Nenhum dado Informado.";
        }

        // Informo ao normalizador se a informação está OK
        $this->normalize = Array (
            "values" => $return
        );

    }

    /**
     * Normaliza os Dados
     *
     * @param Array $rede
     * @return Boolean/String
     */
    private function normalizeResponse($response) {

        $return = true;

        // Verifica se um valor foi informado.
        if(isset($response)) {

            // Checa se o valor de response é um Array
            if(is_array($response)) {

                // Armazena a quantidade de Saídas Esperadas.
                $total_expected_returns = count($response);

                // Armazena a quantidade de Saídas da Rede
                $total_returns = (count($this->estrutura) -1);

                if($total_expected_returns == $this->estrutura[$total_returns]) {

                    // Armazena o index do primeiro Perceptron da Camada de Saída
                    $index_return = (array_sum($this->estrutura) - ($total_expected_returns - 1));
                    
                    // Percorre o Array de Saídas Esperadas
                    for ($i = 0; $i < $total_expected_returns; $i++) {
                        
                        // Armazena os valores das Saídas Esperadas na Rede. 
                        $this->expected_returns[($index_return + $i)] = $response[$i];
                    }

                } else {
                    $return = "A quantidade de Neurônios de Saída não são iguais a quantidade de Saídas Esperadas.";
                }

            } else {
                $return = "Os dados informados não estão armazenados em um Array.";
            }

        } else {
            $return = "Nenhum dado Informado.";
        }

        // Informo ao normalizador se a informação está OK
        $this->normalize = Array (
            "response" => $return
        );
    }

    /**
     * Normaliza os Dados
     *
     * @param Array $rede
     * @return Boolean
     */
    private function normalizeError($error_learn) {

        $return = true;

        // Verifica se outro valor foi informado e se é diferente do padrão.
        if(isset($error_learn) && $error_learn != $this->taxa_aprendizagem) {

            // Armazeno o novo valor.
            $this->taxa_aprendizagem = $error_learn;
        }

        // Informo ao normalizador se a informação está OK
        $this->normalize = Array (
            "error_learn" => $return
        );
    }

    /**
     * Normaliza os Dados
     *
     * @param Array $rede
     * @return Boolean/String
     */
    private function normalizeWeight($weight_variation) {

        $return = true;

        // Verifica se outro valor foi informado e se é diferente do padrão.
        if(isset($weight_variation) && $weight_variation != $this->variation_weight) {

            // Verifica se o valor informado é um Inteiro e se é maior que 0
            if(is_integer($weight_variation) && $weight_variation > 0) {

                // Armazeno o novo valor.
                $this->variation_weight = $weight_variation;

            } else {
                $return = "O valor informado não é do tipo FLOAT ou é menor que 0.";
            }
        }

        // Informo ao normalizador se a informação está OK
        $this->normalize = Array (
            "weight_variation" => $return
        );
    }

    /**
     * Normaliza os Dados
     *
     * @param Array $rede
     * @return Boolean/String
     */
    private function normalizeBias($bias_variation) {

        $return = true;

        // Verifica se outro valor foi informado e se é diferente do padrão.
        if(isset($bias_variation) && $bias_variation != $this->variation_weight) {

            // Verifica se o valor informado é um Inteiro e se é maior que 0
            if(is_integer($bias_variation) && $bias_variation > 0) {

                // Armazeno o novo valor.
                $this->variation_bias = $bias_variation;

            } else {
                $return = "O valor informado não é do tipo INTEGER ou é menor que 0.";
            }
        }

        // Informo ao normalizador se a informação está OK
        $this->normalize = Array (
            "bias_variation" => $return
        );

    }

    /**
     * Normaliza os Dados
     *
     * @param Array $rede
     * @return Boolean/String
     */
    private function normalizeErrorStop($error) {

        $return = true;

        // Verifica se outro valor foi informado e se é diferente do padrão.
        if(isset($error) && $error != $this->stop_error) {

            // Verifica se o valor informado é um Float e se é maior que 0
            if(is_float($error) && $error > 0) {

                // Armazeno o valor.
                $this->stop_error = $error;

            } else {
                $return = "O valor informado não é do tipo FLOAT ou é menor que 0.";
            }
        }

        // Informo ao normalizador se a informação está OK
        $this->normalize = Array (
            "error" => $return
        );

    }

    /**
     * Normaliza os dados de todos os valores para o bom funcionamento
     * da Rede Neural
     *
     * @return Boolean/String
     */
    private function normalizeAllData($type) {

        // Auxiliar para verificar se todos os dados estão corretos e no padrão.
        $continue_execution = true;

        // Armazena as mensagens de Erro.
        $error_message = "";

        // Percorre o Array normalize
        foreach($this->normalize as $key => $value) {

            /**
             * Se o valor informado for diferente de true algum
             * dado está incorreto ou faltando.
             */
            if($value !== true && $type == "train") {
                
                // Define que o treino não pode ser executado.
                $continue_execution = false;

                // Mostra o Erro gerado
                $error_message .= "Error in data '" . $key . "': " . $value . PHP_EOL;
            }

            if($value !== true && $type == "answer") {

                if($key != "response") {
                    
                    // Define que o treino não pode ser executado.
                    $continue_execution = false;
    
                    // Mostra o Erro gerado
                    $error_message .= "Error in data '" . $key . "': " . $value . PHP_EOL;
                }
                
            }
        }

        // Retorna se a execução deve continuar.
        if($continue_execution !== true) {
            return $error_message;
        } else {
            return true;
        }
    }

    // ---- FIM Normalize ----\\

    /**
     * Informa a rede os pesos de cada Sinapse
     * manualmente seguindo o padrão abaixo:
     * 
     * Exemplo de Rede 2x2x1
     * 
     * Os 2 Perceptrons de entrada receberão o nome de 1 e 2.
     * Os 2 Perceptrons da Camada Oculta receberão o nome de 3 e 4 
     * O Perceptron de Saída receberá o nome de 5
     * 
     * Obs: se existirem mais valores o número será sequencial.
     * 
     * A rede ficaria assim:
     * 
     *  1 --- 3
     *    \ /   \
     *     X      5
     *    / \   /
     *  2 --- 4
     * 
     * Então a Sinapse que liga o Perceptron
     * 1 ao Perceptron 3 recebera o nome w1-3
     * logo temos que informar os pesos da
     * seguinte forma
     * 
     * Array(
     *   "w1-3" => 0.15,
     *   "w1-4" => 0.25,
     *
     *   "w2-3" => 0.2,
     *   "w2-4" => 0.3,
     *   
     *   "w3-5" => 0.4,
     *   "w3-5" => 0.5,
     *   
     *   "w4-5" => 0.4,
     *   "w4-5" => 0.5,
     * );
     * 
     * Obs2: Esse array deve ser passado a Rede
     * serialized.
     *
     * @param String $serialize
     * @return Void
     */
    public function setWeight($serialize) {
        $this->pesos = unserialize($serialize);
    }

    /**
     * Informa os Bias de cada camada da Rede
     * lembrando que a camada de Entrada não
     * recebe um valor de Bias.
     * 
     * Segundo o exemplo da nossa Rede 2x2x1
     * Cada coluna representa uma camada da
     * Rede, onde a camada de entrada será 0
     * e a próxima será 1 e assim sucessivamente
     * até a camada de saída.
     * 
     *   _0_       _1_
     *  | 1 | --- | 3 |
     *  |   | \ / |   |\ _2_
     *  |   |  X  |   | | 5 |
     *  |   | / \ |   |/ ---
     *  | 2 | --- | 4 |
     *   ---       ---
     * 
     * Então deve-se informar os Bias da seguinte
     * forma:
     * 
     * Array(
     *   "b1" => 0.5,
     *   "b2" => 0.2,
     * );
     *
     * Obs2: Esse array deve ser passado a Rede
     * serialized.
     * 
     * @param String $serialize
     * @return Void
     */
    public function setBias($serialize) {
        $this->bias = unserialize($serialize);
    }

    /**
     * Salva a resposta de cada trino da rede.
     *
     * @return Void
     */
    private function saveData() {

        // Exporto somente os pesos e bias para salvar no BD
        $response = $this->exportThisTrain();

        // Guarda a chave de cada treino
        $key = "";

        // Concateno todas os valores de entrada para gerar a chave
        foreach($this->entradas as $value) {
            $key .= $value;
        }

        // Armazena a quantidade de Perceptrons na Camada de Saída
        $fim = $this->estrutura[(count($this->estrutura) -1)];

        /**
         * Armazena a quantidade total de Perceptrons da Rede sem
         * contar os Perceptrons da Camada de Saída
         */
        $total = (array_sum($this->estrutura) - $fim);

        // Armazena a soma das saídas.
        $sum = 0;

        // Percorre o Array das Funções de Ativação
        for($i = 1; $i <= $fim; $i++) {

            // Somo todas as saídas
            $sum += $this->activation["y" . ($total + $i)];
        }

        // Salvo a resposta no BD
        $this->train_database[md5($key)] = Array(
            "response" => $response,
            "sum" => $sum
        );

        // Zero os pesos e bias para serem gerados novamente.
        $this->pesos = Array();
        $this->bias = Array();
    }

    /**
     * Inicializador de treinamento da Rede
     *
     * @return Void
     */
    public function train() {

        // Armazena o valor que define se os dados informados estão corretos.
        $execute = $this->normalizeAllData("train");
        
        if($execute === true) {
                
            /**
             * Checa se os Pesos já foram informados ou se
             * precisa gera-los.
             */
            if(!is_array($this->pesos) || count($this->pesos) == 0) {
                $this->setRandWeight();
            }
            
            /**
             * Checa se os Bias já foram informados ou se
             * precisa gera-los.
             */
            if(!is_array($this->bias) || count($this->bias) == 0) {
                $this->setRandBias();
            }
            
            // Conto a quantidade de Camadas Ocultas
            $quantidade_hidden_layers = (count($this->estrutura) - 2);

            // Verifico se existem Camadas Ocultas
            if($quantidade_hidden_layers > 0) {

                // Executo o processo de FeedForward
                $this->feedForward();
                // Calculo o erro de Saída. 
                $this->errorReturn();
                
                // Verifico se o erro de saída é aceitável para parar a execução
                if($this->error["total"] > $this->stop_error) {

                    // Calculo o Gradient das Sinapses da Camada de Saída.
                    $this->calculateLastSinapses();
                    
                    // Calculo o Gradient Descent das Sinapses de cada Camada Oculta
                    for($i = $quantidade_hidden_layers; $i > 0 ; $i--) {
                        $this->calculateSinapsesHiddenLayer($i);
                    }

                    /**
                     * Contador de Épocas, para saber quantas vezes a Rede
                     * executou o FeedForward e Backpropagation.
                     */
                    $this->epoch++;

                    // Reset alguns valores para a rede fazer o calculo novamente.
                    $this->restartEpoch();

                    // Executa um novo treino com os valores atualizados.
                    $this->train();

                } else {
                    $this->saveData();
                }

            }

        } else {
            return $execute;
        }
    }

    public function answerBinary() {

        // Armazena o valor que define se os dados informados estão corretos.
        $execute = $this->normalizeAllData("answer");
                        
        if($execute === true) {

            // Armazena a saída da rede
            $return = array();

            // Armazena a melhor resposta da rede
            $best_response = 1;

            foreach($this->train_database as $value) {

                $is_the_best = false;

                $this->pesos = $value["response"]["weight"];
                $this->bias = $value["response"]["bias"];

                // Executa o FeedForward
                $this->feedForward();

                // Armazena a quantidade de Perceptrons na Camada de Saída
                $fim = $this->estrutura[(count($this->estrutura) -1)];

                /**
                 * Armazena a quantidade total de Perceptrons da Rede sem
                 * contar os Perceptrons da Camada de Saída
                 */
                $total = (array_sum($this->estrutura) - $fim);
                
                $aux = Array();
                $sum = 0;
                $big = 0;

                // Percorre o Array das Funções de Ativação
                for($i = 1; $i <= $fim; $i++) {

                    $exit = $this->activation["y" . ($total + $i)];

                    // Soma todos as saídas
                    $sum += $exit;

                    // Armazena a Saída dos Perceptrons de Saída
                    $aux[] = $exit;

                    if($exit > $big) {
                        $big = $exit;
                    }
                }

                $res = abs($sum - $value["sum"]);

                // Verifica se essa resposta é melhor que a anterior
                if($best_response >= $res) {
                    $best_response = $res;
                    $is_the_best = true;
                }
                
                if($is_the_best) {
                    $return = $aux;
                    //$return["accuracy"] = round((($big * 100) / (1 - $this->stop_error)), 2) . "%";
                    //$return["epoch"] = $this->epoch;
                }
                
            }
            
            // Retorna as Saídas
            return $return;

        } else {
            return $execute;
        }
    }

    /**
     * Informa as entradas da rede.
     *
     * @param Array $entradas
     * @return Array
     */
    public function setValues($entradas) {
        $this->normalizeValues($entradas);
    }

    /**
     * Informa a saída esperada da rede.
     *
     * @param Array $entradas
     * @return Array
     */
    public function setResponse($returns) {
        $this->normalizeResponse($returns);
    }

    /**
     * Gera Pesos aleatórios para as Sinapses da Rede
     * com base na Variação de Peso informada.
     *
     * @return Void
     */
    private function setRandWeight() {

        // Index do Perceptron inicial.
        $perceptrons = 1;

        // Armazena a quantidade de Perceptrons da próxima Camada.
        $start_hide_layer = $this->estrutura[0] + 1;

        // Armazena a quantidade total de Camadas da Rede.
        $quantidade_sinapses = (count($this->estrutura) - 1);

        // Percorre cada Camada da Rede.
        for($i = 0; $i < $quantidade_sinapses; $i++) {

            // Percorre cada Perceptron da Camada atual.
            for($j = 0; $j < $this->estrutura[$i]; $j++) {

                // Percorre cada Perceptron da próxima Camada.
                for($k = 0; $k < $this->estrutura[($i + 1)]; $k++) {

                    // Gera um número aleatório
                    $rand = (mt_rand(($this->variation_weight * -1), $this->variation_weight) / $this->variation_weight);

                    // Executa o calculo randômico para os Pesos das Sinapses.
                    $this->pesos["w" . $perceptrons . "-" . ($k + $start_hide_layer)] = $rand;
                }

                // Passo para o próximo Perceptron da camada atual.
                $perceptrons++;
            }

            // Passo para a próxima Camada.
            $start_hide_layer += $this->estrutura[($i + 1)];
        }
    }

    /**
     * Gera os Bias aleatórios para as Camadas da Rede
     * com base na Variação de Bias informado.
     *
     * @return Void
     */
    private function setRandBias() {

        // Armazena toda a Estrutura da Rede
        $camadas = $this->estrutura;

        // Retira as Entradas da Estrutura
        array_shift($camadas);

        // Conta a quantidade de Camadas.
        $quantidade_camadas = count($camadas);

        // Percorre as Camadas
        for($i = 0; $i < $quantidade_camadas; $i++) {

            $rand = (mt_rand(($this->variation_bias * -1), $this->variation_bias) / $this->variation_bias);

            // Executa o calculo randômico para os Pesos das Camadas.
            $this->bias["b" . ($i + 1)] = $rand;
        }
    }

    /**
     * Executa o FeedForward da Rede
     *
     * @return Void
     */
    private function feedForward() {

        // Pega a quantidade de Perceptrons da primeira Camada Oculta
        $start_hide_layer = $this->estrutura[0] + 1;

        // Armazena a posição do Perceptron utilizado no momento.
        $index_hide_layer = 1;

        // Armazena a quantidade de Sinapses relacionadas com a Camada em execução.
        $quantidade_sinapses = (count($this->estrutura) - 1);

        // Informa a Camada atual sendo processada.
        $camadas = 1;

        /**
         * Variável auxiliar que armazenará os valores dos 
         * Perceptrons após passarem pela função de ativação.
         */
        $aux_entradas = Array();

        // Armazena as Entradas da Rede em uma variável auxiliar.
        $entradas = $this->entradas;

        // Percorre todas as Sinapses relacionadas com a Camada em execução.
        for($i = 0; $i < $quantidade_sinapses; $i++) {

            // Percorre todos os Perceptrons relacionados com a Camada em execução.
            for($j = 0; $j < $this->estrutura[($i + 1)]; $j++) {

                // Popula o Array de Perceptrons com o valor 0.
                $this->perceptrons["y" . ($start_hide_layer + $j)] = 0;

                /** 
                 * Popula o Array que contém os valores dos 
                 * Perceptrons Ativados com o valor 0.
                 */ 
                $this->activation["y" . ($start_hide_layer + $j)] = 0;

                // Percorre o Array de Estrutura na Camada anterior a de Execução.
                for($k = 0; $k < $this->estrutura[$i]; $k++) {

                    /**
                     * Armazena a multiplicação o Valor do Perceptron atual 
                     * com o valor da Sinapse interligada a ele relativa ao
                     * Perceptron atual da Camada Oculta.
                     */
                    $operation = $entradas[$k] * $this->pesos["w" . ($k + $index_hide_layer) . "-" . ($start_hide_layer + $j)];

                    /**
                     * Armazena a informação anterior somando-a com o valor já
                     * existente no Array de Perceptrons.
                     */
                    $this->perceptrons["y" . ($start_hide_layer + $j)] += $operation;
                }

                /**
                 * Soma o valor final do Perceptron atual da Camada Oculta
                 * com o Bias da Camada Atual.
                 */
                $this->perceptrons["y" . ($start_hide_layer + $j)] += ($this->bias["b" . $camadas] * 1);

                // Armazena o valor do Perceptron após ser Ativado no Array de Ativação.
                $this->activation["y" . ($start_hide_layer + $j)] = $this->sigmoide($this->perceptrons["y" . ($start_hide_layer + $j)]);

                // Armazena o resultado da Ativação no Array auxiliar.
                $aux_entradas[$j] = $this->activation["y" . ($start_hide_layer + $j)];
            }

            /**
             * Define que as entradas agora serão os valores calculados
             * atualmente da Camada Oculta para seguir com o FeedForward
             */
            $entradas = $aux_entradas;

            // Atribui que a Rede deve calcular a próxima Camada.
            $camadas++;

            // Redefine o index inicial da próxima Camada.
            $index_hide_layer = $start_hide_layer;

            // Redefine a quantidade de Perceptrons da próxima Camada a ser executada.
            $start_hide_layer += $this->estrutura[($i + 1)];
        }
    }

    /**
     * Função de Ativação
     *
     * @param Float $number
     * @return Float
     */
    private function sigmoide($number) {
        return (1 / (1 + exp(-$number)));
    }

    /**
     * Derivada da Função de Ativação
     *
     * @param Float $number
     * @return Float
     */
    private function derivada_sigmoide($number) {
        return ($number * (1 - $number));
    }

    /**
     * Calcula o Erro de Saída Total encontrado na
     * Camada de Saída com base nas Saídas Esperadas.
     *
     * @return Void
     */
    private function errorReturn() {

        // Armazena a quantidade de Saídas Esperadas
        $total_returns = count($this->expected_returns);

        // Armazena o index do Perceptron de Saída a ser calculado.
        $index_return = (array_sum($this->estrutura) - ($total_returns - 1));

        // Popula o Erro Total com o valor 0.
        $total = 0;

        // Percorre todos os Perceptrons de Saída
        for($i = 0; $i < $total_returns; $i++) {

            /**
             * Calcula o Erro desse Perceptron:
             * 
             * ((1/2) * ((Saída Esperada Relativa ao Perceptron atual) - Saída Obtida do Perceptron atual) Elevado a 2).
             */
            $operation = ((1 / 2) * (($this->expected_returns[($index_return + $i)] - $this->activation["y" . ($index_return + $i)]) ** 2));

            // Armazena o Erro desse Perceptron.
            $this->error["e" . ($index_return + $i)] = $operation;

            // Soma o Erro desse Perceptron com os valores anteriores.
            $total += $operation;
        }

        // Armazena o Erro Total.
        $this->error["total"] = $total; 
    }

    /**
     * Calcula o quanto o valor da Sinapse atual
     * contribuiu para o Erro Total.
     *
     * @param Float $weight
     * @param Float $gradient
     * @return Float
     */
    private function errorAdjust($weight, $gradient) {
        return ($weight - ($this->taxa_aprendizagem * $gradient));
    }

    /**
     * Calcula o Gradient Descent da Camada de Saída
     *
     * @param Integer $w_id
     * @return Void
     */
    private function calculateLastGradientDescent($w_id) {

        // Percorre o Array de Pesos
        foreach($this->pesos as $key => $value) {

            // Verifica se no Peso atual existe o index do Perceptron ($w_id).
            if(preg_match("/w[0-9]*-" . $w_id . "/", $key)) {

                // Calcula a Derivada do Erro em relação ao Perceptron atual.
                $derivadaDoErroSobreOPerceptron = (($this->expected_returns[$w_id] * -1) + $this->activation["y" . $w_id]);

                // Calcula a Derivada da Função de Ativação do Perceptron atual.
                $derivationFunctionActivation = $this->derivada_sigmoide($this->activation["y" . $w_id]);

                // Retorna o valor do Perceptron da Camada anterior interligado ao Perceptron atual
                preg_match("/w([0-9]*)-" . $w_id . "/", $key, $perceptronAnterior);

                // Calcula a Derivada do Perceptron relativo a Sinapse interligada a ele.
                $derivadaDoPerceptronSobreASinapse = $this->activation["y" . $perceptronAnterior[1]];

                /**
                 * Calcula o Gradient Descent da Derivada da Função de Ativação
                 * do Perceptron atual multiplicado pela Derivada do Perceptron
                 * relativo a Sinapse ligada a ele.
                 */
                $gradient = $derivadaDoErroSobreOPerceptron * $derivationFunctionActivation * $derivadaDoPerceptronSobreASinapse;

                /**
                 * Armazena o valor da Derivada do Erro relativo ao Perceptron atual
                 * multiplicado pela Derivada da Função de Ativação do Perceptron atual
                 * multiplicado pelo Peso da Sinapse interligada ao Perceptron atual. 
                 */ 
                $this->gradient["g" . $perceptronAnterior[1] . "-" . $w_id] = Array(
                    "D(E(Y))/D(F(Y))" => (($derivadaDoErroSobreOPerceptron * $derivationFunctionActivation) * $this->pesos[$key]),
                );

                // Armazena o valor do Erro da Sinapse atual relativo ao Erro Total.
                $adjust = $this->errorAdjust($value, $gradient);

                // Armazena o novo peso da Sinapse atual relativa ao Perceptron atual.
                $this->novos_pesos[$key] = $adjust;
            }
        }
    }

    /**
     * Percorre as Sinapses interligadas aos Perceptrons da Camada de Saída.
     *
     * @return Void
     */
    private function calculateLastSinapses() {

        // Armazena a quantidade de Sinapses interligadas aos Perceptrons de Saída.
        $sinapses_atualizar = (array_sum($this->estrutura) - $this->estrutura[(count($this->estrutura) - 1)] +1);

        // Armazena o total de Sinapses da Rede
        $total_sinapses = $sinapses_atualizar + $this->estrutura[(count($this->estrutura) - 1)] -1;

        // Informa apenas as Sinapses que estão interligadas aos Perceptrons de Saída. 
        for ($i = $sinapses_atualizar; $i <= $total_sinapses; $i++) {

            // Calcula o novo Peso da Sinapse informada.
            $this->calculateLastGradientDescent($i);
        }
    }

    /**
     * Calcula o Gradient Descent de todas as Camadas Ocultas
     *
     * @param Integer $w_id
     * @param Integer $index
     * @return Void
     */
    private function calculateHiddenGradientDescent($w_id, $index) {

        // Percorre o Array de Pesos
        foreach($this->pesos as $key => $value) {

            // Verifica se no Peso atual existe o index do Perceptron ($w_id).
            if(preg_match("/w[0-9]*-" . $w_id . "/", $key)) {

                // Retorna o valor do Perceptron da Camada anterior interligado ao Perceptron atual
                preg_match("/w([0-9]*)-" . $w_id . "/", $key, $perceptronAnterior);
                
                // Armazena a quantidade de Perceptrons da Camada seguinte.
                $next_perceptrons = $this->estrutura[($index +1)];

                // Popula o index de Saída com o valor 0.
                $index_return = 0;

                // Percorre as Camadas da Rede.
                for($i = 0; $i <= $index; $i++) {

                    // Soma a quantidade de Perceptrons de cada Camada com o valor já existente.
                    $index_return += $this->estrutura[$i];
                }

                // Armazena o index do último Perceptron da Camada atual.
                $index_fim = ($index_return + $next_perceptrons);
                
                /**
                 * Popula a Derivada do Erro Total em relação a Função
                 * de Ativação do Perceptron com o valor 0.
                 */
                $derivadaErroTotalSobreFaDoPerceptron = 0;

                // Percorre o Array de Gradient da Camada seguinte.
                for($i = ($index_return + 1); $i <= $index_fim; $i++) {

                    /**
                     * Soma a Derivada do Erro Total em relação a Função
                     * de Ativação do Perceptron atual com o valor pré
                     * existente.
                     */
                    $derivadaErroTotalSobreFaDoPerceptron += $this->gradient["g" . $w_id . "-" . $i]["D(E(Y))/D(F(Y))"];
                }

                // Armazena a Derivada da Função de Ativação do Perceptron atual.
                $derivadaDaFaDoPerceptron = $this->derivada_sigmoide($this->activation["y" . $w_id]);

                // Armazena a Derivada do Perceptron atual relativo a Sinapse
                $derivadaDoPerceptronsSobreASinapse = $this->activation["y" . $w_id];

                /**
                 * Calcula o Gradient Descent da Derivada do Erro Total em relação a Função
                 * de Ativação do Perceptron atual multiplicado pela Derivada da Função de
                 * Ativação do Perceptron atual multiplicado pela Derivada do Perceptron
                 * relativo a Sinapse.
                 */
                $gradient = $derivadaErroTotalSobreFaDoPerceptron * $derivadaDaFaDoPerceptron * $derivadaDoPerceptronsSobreASinapse;

                // Armazena o Gradient Descent da Sinapse atual
                $this->gradient["g" . $perceptronAnterior[1] . "-" . $w_id] = Array(

                    /**
                     * Sendo ele a multiplicação da Derivada do Erro Total relativo
                     * a Função de Ativação do Perceptron pela Derivada da Função de
                     * Ativação do Perceptron pelo Peso da Sinapse atual.
                     */
                    "D(E(Y))/D(F(Y))" => $derivadaErroTotalSobreFaDoPerceptron * $derivadaDaFaDoPerceptron * $this->pesos[$key],
                );

                // Armazena o valor do Erro da Sinapse atual relativo ao Erro Total.
                $adjust = $this->errorAdjust($value, $gradient);

                // Armazena o novo peso da Sinapse atual relativa ao Perceptron atual.
                $this->novos_pesos[$key] = $adjust;
            }
        }
    }

    /**
     * Percorre as Sinapses interligadas aos Perceptrons da Camada seguinte.
     *
     * @param Integer $index
     * @return Void
     */
    private function calculateSinapsesHiddenLayer($index) {

        // Popula a quantidade de Perceptrons da Camada Oculta atual com o valor 0.
        $perceptrons_hidden_layers = 0;

        // Percorre as Camadas da Rede
        for($i = 0; $i < $index; $i++) {

            // Soma a quantidade de Perceptrons da Camada atual com o valor pré existente.
            $perceptrons_hidden_layers += $this->estrutura[($i)];
        }

        // Informo as Sinapses relativas a Camada atual
        for ($i = ($perceptrons_hidden_layers +1); $i <= ($perceptrons_hidden_layers + $this->estrutura[$index]); $i++) {

            // Calcula o novo Peso da Sinapse informada.
            $this->calculateHiddenGradientDescent($i,  $index);
        }
    }

    /**
     * Exporta os dados necessários para armazenagem.
     *
     * @return Void
     */
    private function exportThisTrain() {

        // Armazena os dados já treinados pela Rede
        $export = Array(

            // Pesos já reduzidos.
            "weight" => $this->pesos,

            // Bias Gerados
            "bias" => $this->bias
        );

        return $export;
    }

    /**
     * Exporta os dados necessários para executar a Rede
     * numa próxima vez sem precisar treina-la.
     *
     * @return Void
     */
    public function exportData() {

        // Armazena os dados já treinados pela Rede
        $export = Array(

            // Estrutura
            "rede" => $this->rede,

            // Database
            "database" => $this->train_database,

            // Épocas
            "epoch" => $this->epoch,
        );

        return serialize($export);
    }

    /**
     * Importa os dados para a Rede
     *
     * @param String $train
     * @return Void
     */
    public function importData($train) {

        // Reset da Rede
        $this->resetRede();

        // Armazena os dados de treino da Rede
        $import = unserialize($train);

        // Verifica se existem os dados das Estruturas.
        if(isset($import["rede"])) {
            $this->setConfiguration($import["rede"]);
        }

        if(isset($import["database"])) {
            $this->train_database = $import["database"];
        }

        // Verifica se existem os dados dos Epoch.
        if(isset($import["epoch"])) {
            $this->epoch = $import["epoch"];
        }
    }

    /**
     * Prepara os valores da Rede para a próxima Época.
     *
     * @return Void
     */
    private function restartEpoch() {
        
        // Define os Pesos calculados no Backpropagation como os Pesos atuais. 
        $this->pesos = $this->novos_pesos;

        // Zero o Array que continha os Pesos calculados pela Rede.
        $this->novos_pesos = Array();

        // Zero o Array de Gradient calculado pela Rede.
        $this->gradient = Array();

        // Zero o Erro Total calculado pela Rede.
        $this->error = Array();

        // Zero o Array de Perceptrons da Rede.
        $this->perceptrons = Array();
    }

    /**
     * Reset de toda a Rede para o padrão
     *
     * @return Void
     */
    private function resetRede() {
        $this->normalize = Array();
        $this->estrutura = Array();
        $this->entradas = Array();
        $this->expected_returns = Array();
        $this->taxa_aprendizagem = 0.01;
        $this->variation_weight = 100;
        $this->variation_bias = 100;
        $this->stop_error = 0.001;
        $this->perceptrons = Array();
        $this->pesos = Array();
        $this->novos_pesos = Array();
        $this->bias = Array();
        $this->activation = Array();
        $this->gradient = Array();
        $this->error = Array();
        $this->memory_limit = "";
        $this->rede = Array();
        $this->train_database = Array();
        $this->epoch = 0;
    }

    /**
     * Função utilizada para Debug dos dados da Rede
     *
     * @return Void
     */
    public function debug() {
        echo "<pre>";
        
        echo "Erro Total: \n"; 
        print_r($this->error);

        echo "Função de Ativação: ";
        print_r($this->activation);

        echo "Perceptrons: ";
        print_r($this->perceptrons);

        echo "Bias: ";
        print_r($this->bias);

        echo "Pesos: ";
        print_r($this->pesos);

        echo "Novos Pesos: ";
        print_r($this->novos_pesos);

        echo "Gradient Descent: ";
        print_r($this->gradient);

        echo "Entradas: ";
        print_r($this->entradas);

        echo "Saídas Esperadas: ";
        print_r($this->expected_returns);

        echo "Memória Alocada: ";
        print_r($this->memory_limit);

        echo "Configurações: ";
        print_r($this->rede);

        echo "Database: ";
        print_r($this->train_database);

        echo "Épocas: ";
        print_r($this->epoch);

        echo "</pre>";
    }

}

?>