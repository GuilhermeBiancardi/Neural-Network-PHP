# Neural Network PHP

Uma biblioteca completa de Redes Neurais Artificiais implementada em PHP puro, com suporte a múltiplas arquiteturas, otimizadores e **aceleração por GPU**.

[![PHP Version](https://img.shields.io/badge/PHP-8.1%2B-blue)](https://www.php.net/)
[![GPU Support](https://img.shields.io/badge/GPU-OpenCL-green)](GPU_SETUP.md)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Índice

- [Características](#-características)
- [Aceleração por GPU](#-aceleração-por-gpu)
- [Requisitos](#-requisitos)
- [Instalação](#-instalação)
- [Uso Básico](#-uso-básico)
- [Componentes](#-componentes)
  - [Camadas (Layers)](#camadas-layers)
  - [Funções de Ativação](#funções-de-ativação)
  - [Otimizadores](#otimizadores)
  - [Funções de Perda (Loss)](#funções-de-perda-loss)
- [Configuração Avançada](#-configuração-avançada)
- [Exemplos Práticos](#-exemplos-práticos)
- [Salvamento e Carregamento de Modelos](#-salvamento-e-carregamento-de-modelos)
- [API Completa](#-api-completa)
- [Integração com php_tensor](#-integração-com-php_tensor)
- [Contribuindo](#-contribuindo)
- [Licença](#-licença)

## Características

- **Aceleração por GPU**: Suporte a OpenCL para operações 20x a 50x mais rápidas
- **Múltiplas Camadas**: Dense, Conv2D, Dropout, BatchNormalization, Flatten
- **Funções de Ativação**: Sigmoid, ReLU, LeakyReLU, ELU, Tanh, Softmax, Linear
- **Otimizadores Modernos**: SGD, Adam, AdamW, RMSProp
- **Funções de Perda**: MSE (Mean Squared Error), CrossEntropy
- **Regularização**: Dropout, BatchNormalization, Weight Decay, Gradient Clipping
- **Early Stopping**: Parada antecipada para evitar overfitting
- **Mini-Batch Training**: Treinamento eficiente com lotes
- **Salvamento/Carregamento**: Persistência completa de modelos treinados
- **Detecção Automática de Backend**: GPU → CPU otimizado → PHP puro

## Como Habilitar a Aceleração por GPU

Esta biblioteca suporta **aceleração por GPU via OpenCL**, oferecendo ganhos de performance de **20x a 50x** para operações matriciais grandes!

Veja [GPU_SETUP.md](GPU_SETUP.md) para instruções detalhadas de instalação (Windows e Linux).

A biblioteca **detecta automaticamente** se a GPU está disponível e a usa quando possível. Nenhuma mudança no código é necessária, recomendo fazer o benchmark para verificar a performance.

## Como Habilitar a Aceleração por CPU

Esta biblioteca suporta **aceleração por CPU via tensor**, oferecendo ganhos de performance de **2x a 5x** para operações matriciais grandes!

```bash
# Instalar tensor
pecl install tensor
# Ou baixe manualmente copiando a dll php_tensor.dll para a pasta ext do php
# e a Dll libopenblas.dll para a pasta raíz do php depois habilite no php.ini
# adicionando extension=tensor
```

A biblioteca **detecta automaticamente** se a CPU está disponível e a usa quando possível. Nenhuma mudança no código é necessária, recomendo fazer o benchmark para verificar a performance.

### Se preferir usar o PHP puro

Se preferir usar o PHP puro, você não precisa instalar nada extra.

### Performance Comparada

| Operação | Tamanho | PHP Puro | GPU (OpenCL) | Speedup |
|----------|---------|----------|--------------|---------|
| Matrix Multiply | 100x100 | 5.2 ms | 0.4 ms | **13x**  |
| Matrix Multiply | 500x500 | 650 ms | 15 ms | **43x**   |
| Matrix Multiply | 1000x1000 | 5200 ms | 95 ms | **55x**   |

```php
<?php
use NeuralNetwork\Helper\Matrix;

// Verificar se GPU está disponível
if (Matrix::isGpuAvailable()) {
    echo "GPU Acceleration enabled!\n";
}

// Obter informações do backend
$info = Matrix::getBackendInfo();
echo "Using: " . $info['description'] . "\n";
```

**Benchmark**: Execute `php examples/benchmark_gpu.php` para testar a performance no seu sistema.

## Requisitos

- **PHP >= 8.3 && < 8.4 se for usar o tensor ou PHP 8.3 superior caso contrário**
- **Opcional (GPU)**: Rindow Math Matrix + OpenCL drivers
- **Opcional (CPU)**: Extensão `tensor` para aceleração CPU

## Instalação

Clone o repositório e inclua a classe principal no seu projeto:

```php
require_once 'NeuralNetwork.class.php';
```

## Uso Básico

### Exemplo: Problema XOR

```php
<?php
require_once 'NeuralNetwork.class.php';

use NeuralNetwork\Layer\Dense;

// Dataset XOR
$inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
];
$targets = [
    [0],
    [1],
    [1],
    [0],
];

// Construir a rede: 2 entradas -> 8 neurônios ocultos -> 1 saída
$layers = [
    new Dense(2, 8, 'relu'),
    new Dense(8, 1, 'sigmoid'),
];

$nn = new NeuralNetwork($layers);
$nn->configure([
    'learning_rate' => 0.01,
    'optimizer' => 'adam',
    'loss' => 'mse',
]);

// Treinar
$nn->train($inputs, $targets, 10000, 0, true);

// Fazer predições
foreach ($inputs as $input) {
    $output = $nn->predict($input);
    echo "Entrada: [" . implode(', ', $input) . "] => Predição: " . round($output[0]) . "\n";
}
```

## Componentes

### Camadas (Layers)

#### Dense (Camada Totalmente Conectada)

Camada densa tradicional onde cada neurônio está conectado a todos os neurônios da camada anterior.

```php
use NeuralNetwork\Layer\Dense;

// Dense(entradas, saídas, função_ativação)
$layer = new Dense(10, 20, 'relu');
```

**Parâmetros:**
- `$inputSize` (int): Número de entradas
- `$outputSize` (int): Número de neurônios/saídas
- `$activation` (string): Função de ativação ('relu', 'sigmoid', 'tanh', 'softmax', 'linear', 'leakyrelu', 'elu')

#### Conv2D (Camada Convolucional 2D)

Camada convolucional para processamento de imagens e dados espaciais.

```php
use NeuralNetwork\Layer\Conv2D;

// Conv2D(filtros, kernel_size, stride, padding, ativação)
$layer = new Conv2D(32, 3, 1, 1, 'relu');
```

**Parâmetros:**
- `$filters` (int): Número de filtros/kernels
- `$kernelSize` (int): Tamanho do kernel (ex: 3 para 3x3)
- `$stride` (int): Passo da convolução
- `$padding` (int): Padding ao redor da entrada
- `$activation` (string): Função de ativação

#### Dropout

Regularização que desativa aleatoriamente neurônios durante o treinamento para prevenir overfitting.

```php
use NeuralNetwork\Layer\Dropout;

// Dropout(taxa_de_dropout)
$layer = new Dropout(0.5); // Desativa 50% dos neurônios
```

**Parâmetros:**
- `$rate` (float): Taxa de dropout (0.0 a 1.0)

#### BatchNormalization

Normaliza as ativações de uma camada para acelerar o treinamento e melhorar a estabilidade.

```php
use NeuralNetwork\Layer\BatchNormalization;

// BatchNormalization(número_de_features)
$layer = new BatchNormalization(64);
```

**Parâmetros:**
- `$numFeatures` (int): Número de features a normalizar

#### Flatten

Achata tensores multidimensionais em vetores 1D, útil entre camadas convolucionais e densas.

```php
use NeuralNetwork\Layer\Flatten;

$layer = new Flatten();
```

### Funções de Ativação

| Função | Descrição | Uso Recomendado |
|--------|-----------|-----------------|
| **sigmoid** | Saída entre 0 e 1 | Classificação binária (camada de saída) |
| **relu** | max(0, x) | Camadas ocultas (padrão) |
| **leakyrelu** | max(0.01x, x) | Camadas ocultas (evita neurônios mortos) |
| **elu** | Exponential Linear Unit | Camadas ocultas (convergência mais rápida) |
| **tanh** | Saída entre -1 e 1 | Camadas ocultas (dados centrados em zero) |
| **softmax** | Distribuição de probabilidade | Classificação multiclasse (camada de saída) |
| **linear** | Identidade (sem transformação) | Regressão (camada de saída) |

### Otimizadores

#### SGD (Stochastic Gradient Descent)

Otimizador básico com suporte a momentum, weight decay e gradient clipping.

```php
$nn->configure([
    'optimizer' => 'sgd',
    'learning_rate' => 0.01,
    'momentum' => 0.9,
    'weight_decay' => 0.0001,
    'gradient_clip' => 5.0,
]);
```

#### Adam (Adaptive Moment Estimation)

Otimizador adaptativo que combina momentum e RMSProp. Excelente escolha padrão.

```php
$nn->configure([
    'optimizer' => 'adam',
    'learning_rate' => 0.001,
]);
```

#### AdamW (Adam with Weight Decay)

Variante do Adam com weight decay desacoplado, melhor para regularização.

```php
$nn->configure([
    'optimizer' => 'adamw',
    'learning_rate' => 0.001,
    'weight_decay' => 0.01,
    'gradient_clip' => 1.0,
]);
```

#### RMSProp

Otimizador adaptativo que funciona bem com redes recorrentes.

```php
$nn->configure([
    'optimizer' => 'rmsprop',
    'learning_rate' => 0.001,
]);
```

### Funções de Perda (Loss)

#### MSE (Mean Squared Error)

Para problemas de regressão.

```php
$nn->configure(['loss' => 'mse']);
```

#### CrossEntropy

Para problemas de classificação.

```php
$nn->configure(['loss' => 'crossentropy']);
```

## ⚙️ Configuração Avançada

### Exemplo Completo com Todas as Opções

```php
use NeuralNetwork\Layer\Dense;
use NeuralNetwork\Layer\BatchNormalization;
use NeuralNetwork\Layer\Dropout;

$layers = [
    new Dense(10, 64, 'relu'),
    new BatchNormalization(64),
    new Dropout(0.3),
    new Dense(64, 32, 'relu'),
    new Dropout(0.2),
    new Dense(32, 1, 'linear'),
];

$nn = new NeuralNetwork($layers);
$nn->configure([
    'learning_rate' => 0.001,      // Taxa de aprendizado
    'optimizer' => 'adamw',         // Otimizador
    'loss' => 'mse',                // Função de perda
    'batch_size' => 32,             // Tamanho do lote
    'momentum' => 0.9,              // Momentum (para SGD)
    'weight_decay' => 0.01,         // Regularização L2
    'gradient_clip' => 5.0,         // Gradient clipping
]);

// Treinar com early stopping
$nn->train(
    $inputs,
    $targets,
    epochs: 1000,
    patience: 50,      // Para após 50 épocas sem melhoria
    verbose: true      // Mostrar progresso
);
```

## Exemplos Práticos

### 1. Classificação Binária (XOR)

```php
$inputs = [[0,0], [0,1], [1,0], [1,1]];
$targets = [[0], [1], [1], [0]];

$layers = [
    new Dense(2, 8, 'relu'),
    new Dense(8, 1, 'sigmoid'),
];

$nn = new NeuralNetwork($layers);
$nn->configure(['optimizer' => 'adam', 'loss' => 'mse']);
$nn->train($inputs, $targets, 10000);
```

### 2. Regressão (Função Seno)

```php
$inputs = [];
$targets = [];
for ($i = 0; $i < 100; $i++) {
    $x = $i / 10.0;
    $inputs[] = [$x];
    $targets[] = [sin($x)];
}

$layers = [
    new Dense(1, 32, 'relu'),
    new Dense(32, 32, 'tanh'),
    new Dense(32, 1, 'linear'),
];

$nn = new NeuralNetwork($layers);
$nn->configure(['optimizer' => 'adam', 'loss' => 'mse']);
$nn->train($inputs, $targets, 5000, 0, true);
```

### 3. Classificação Multiclasse

```php
// Dataset Iris simplificado
$inputs = [
    [5.1, 3.5, 1.4, 0.2], // Setosa
    [7.0, 3.2, 4.7, 1.4], // Versicolor
    [6.3, 3.3, 6.0, 2.5], // Virginica
    // ... mais exemplos
];

$targets = [
    [1, 0, 0], // Setosa
    [0, 1, 0], // Versicolor
    [0, 0, 1], // Virginica
    // ... mais exemplos
];

$layers = [
    new Dense(4, 16, 'relu'),
    new Dense(16, 8, 'relu'),
    new Dense(8, 3, 'softmax'),
];

$nn = new NeuralNetwork($layers);
$nn->configure(['optimizer' => 'adam', 'loss' => 'crossentropy']);
$nn->train($inputs, $targets, 1000, 0, true);

// Predição
$prediction = $nn->predict([5.0, 3.0, 1.6, 0.2]);
// Resultado: [0.95, 0.03, 0.02] -> Classe 0 (Setosa)
```

### 4. Rede Convolucional (CNN)

```php
use NeuralNetwork\Layer\Conv2D;
use NeuralNetwork\Layer\Flatten;

$layers = [
    new Conv2D(32, 3, 1, 1, 'relu'),  // 32 filtros 3x3
    new Conv2D(64, 3, 1, 1, 'relu'),  // 64 filtros 3x3
    new Flatten(),
    new Dense(64 * 28 * 28, 128, 'relu'),
    new Dropout(0.5),
    new Dense(128, 10, 'softmax'),
];

$nn = new NeuralNetwork($layers);
$nn->configure([
    'optimizer' => 'adam',
    'loss' => 'crossentropy',
    'batch_size' => 64,
]);
```

## Salvamento e Carregamento de Modelos

### Salvar Modelo Treinado

```php
// Treinar o modelo
$nn->train($inputs, $targets, 1000);

// Salvar em arquivo
$nn->save('modelo_treinado.bin');
```

### Carregar Modelo

```php
// Carregar modelo previamente treinado
$nn = NeuralNetwork::load('modelo_treinado.bin');

// Usar diretamente para predições
$prediction = $nn->predict([1.5, 2.3]);
```

**Nota:** O salvamento preserva:
- Arquitetura completa da rede (camadas)
- Pesos e biases treinados
- Estado do otimizador
- Configuração da função de perda

## API Completa

### Construtor

```php
public function __construct(array $layers, float $learningRate = 0.01)
```

**Parâmetros:**
- `$layers`: Array de objetos `LayerInterface`
- `$learningRate`: Taxa de aprendizado inicial (padrão: 0.01)

### configure()

```php
public function configure(array $options): void
```

**Opções disponíveis:**
- `learning_rate` (float): Taxa de aprendizado
- `optimizer` (string): 'sgd', 'adam', 'adamw', 'rmsprop'
- `loss` (string): 'mse', 'crossentropy'
- `batch_size` (int): Tamanho do mini-batch (padrão: 32)
- `momentum` (float): Momentum para SGD (padrão: 0.0)
- `weight_decay` (float): Regularização L2 (padrão: 0.0)
- `gradient_clip` (float): Valor máximo do gradiente (padrão: 0.0 = desabilitado)

### train()

```php
public function train(
    array $inputs,
    array $targets,
    int $epochs,
    int $patience = 0,
    bool $verbose = false
): void
```

**Parâmetros:**
- `$inputs`: Array de amostras de entrada
- `$targets`: Array de valores alvo correspondentes
- `$epochs`: Número de épocas de treinamento
- `$patience`: Épocas sem melhoria antes de parar (0 = desabilitado)
- `$verbose`: Exibir progresso do treinamento

**Formato dos dados:**
```php
// Entrada única
$inputs = [[x1, x2, x3]];
$targets = [[y1, y2]];

// Múltiplas entradas
$inputs = [
    [x1, x2, x3],
    [x1, x2, x3],
    // ...
];
$targets = [
    [y1, y2],
    [y1, y2],
    // ...
];
```

### predict()

```php
public function predict(array $input): array
```

**Parâmetros:**
- `$input`: Amostra única ou lote de amostras

**Retorno:**
- Array com as predições

**Exemplos:**
```php
// Predição única
$output = $nn->predict([1.0, 2.0, 3.0]);
// Retorna: [0.85, 0.15]

// Predição em lote
$outputs = $nn->predict([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
]);
// Retorna: [[0.85, 0.15], [0.23, 0.77]]
```

### save()

```php
public function save(string $filepath): void
```

**Parâmetros:**
- `$filepath`: Caminho do arquivo para salvar o modelo

### load() (estático)

```php
public static function load(string $filepath): NeuralNetwork
```

**Parâmetros:**
- `$filepath`: Caminho do arquivo do modelo salvo

**Retorno:**
- Instância de `NeuralNetwork` com o modelo carregado

## Integração com php_tensor

Esta biblioteca suporta a extensão `php_tensor` para aceleração de operações matriciais. Se a extensão estiver instalada, as operações serão automaticamente aceleradas.

### Verificar Suporte

```php
if (extension_loaded('tensor')) {
    echo "php_tensor está disponível - usando aceleração!\n";
} else {
    echo "Usando implementação PHP pura\n";
}
```

### Instalação do php_tensor

Consulte a documentação oficial do php_tensor para instruções de instalação específicas do seu sistema operacional.

## Dicas de Uso

### Escolhendo a Taxa de Aprendizado

- **Muito alta**: Modelo não converge, loss oscila
- **Muito baixa**: Treinamento muito lento
- **Recomendado**: 
  - SGD: 0.01 - 0.1
  - Adam/AdamW: 0.001 - 0.01
  - RMSProp: 0.001 - 0.01

### Prevenindo Overfitting

1. **Dropout**: Adicione camadas de Dropout (0.2 - 0.5)
2. **Weight Decay**: Use regularização L2 (0.0001 - 0.01)
3. **Early Stopping**: Configure `patience` no treinamento
4. **Batch Normalization**: Adicione entre camadas densas
5. **Mais dados**: Aumente o dataset de treinamento

### Normalização de Dados

Sempre normalize seus dados de entrada:

```php
// Min-Max Normalization (0-1)
function normalize($value, $min, $max) {
    return ($value - $min) / ($max - $min);
}

// Standardization (média 0, desvio 1)
function standardize($value, $mean, $std) {
    return ($value - $mean) / $std;
}
```

### Debugging

```php
// Ativar verbose para ver o progresso
$nn->train($inputs, $targets, 1000, 0, verbose: true);

// Verificar predições durante o treinamento
for ($epoch = 0; $epoch < 100; $epoch++) {
    $nn->train($inputs, $targets, 10, 0, false);
    
    $testPred = $nn->predict($testInput);
    echo "Época $epoch: Predição = " . $testPred[0] . "\n";
}
```
