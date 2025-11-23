<?php
declare (strict_types = 1);

namespace NeuralNetwork\Data;

class DataLoader {
    private array $inputs;
    private array $targets;
    private int $batchSize;
    private bool $shuffle;
    private array $indices;

    public function __construct(array $inputs, array $targets, int $batchSize = 32, bool $shuffle = true) {
        if (count($inputs) !== count($targets)) {
            throw new \InvalidArgumentException("Inputs and targets must have the same length.");
        }
        $this->inputs = $inputs;
        $this->targets = $targets;
        $this->batchSize = $batchSize;
        $this->shuffle = $shuffle;
        $this->indices = range(0, count($inputs) - 1);

        if ($this->shuffle) {
            shuffle($this->indices);
        }
    }

    public function getIterator(): \Generator {
        $numSamples = count($this->inputs);

        for ($start = 0; $start < $numSamples; $start += $this->batchSize) {
            $batchIndices = array_slice($this->indices, $start, $this->batchSize);

            $batchInputs = [];
            $batchTargets = [];
            foreach ($batchIndices as $idx) {
                $batchInputs[] = $this->inputs[$idx];
                $batchTargets[] = $this->targets[$idx];
            }

            yield [$batchInputs, $batchTargets];
        }

        // Reshuffle for next epoch if needed (caller should create new iterator or we reset)
        // Generators cannot be rewound easily if we want to reshuffle.
        // Usually DataLoader is re-instantiated or we have a reset method.
        // But PHP generators are one-time use.
        // So the user should call getIterator() again?
        // If we want to reshuffle, we should do it at the start of getIterator if we could.
        // But constructor did it.
    }

    public function shuffle(): void {
        if ($this->shuffle) {
            shuffle($this->indices);
        }
    }

    public function count(): int {
        return (int) ceil(count($this->inputs) / $this->batchSize);
    }
}
