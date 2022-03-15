# Entropy-based Stability-Plasticity for Lifelong Learning

This repository contains the implementation of ESP.

## Requirements

* torch                1.10.0
* transformers         3.0.2

## Natural Language Experiment

Run the following command to train the model:

```bash
$ bash run_nlp.sh
```

To get it to work, it is mandatory to update the `DATA_DIR` and `OUTPUT_DIR` directories of the `run_nlp.sh` file. Also, it is possible to choose the execution order by setting the `ORDER` variable.
On the other hand, you can modify some parameters for ESP training.
* `mem_capacity`: replay set size from 0 to 1
* `only_mem`: activate to use ONLY setup (See Section 4.1 of the paper)
* `dynamic_freeze`: allow freezing layers on-the-fly 
