# pytorch-template
A scalable PyTorch template for any differentiable graph computation project.

## Overview
```
configs/
  your_experiment_configuration_file.yml
lib/
  trainers/
    coco_trainer.py
  datasets/
    coco.py
    cifar.py
  utils/
    logger.py
  models/
    resnet.py
    segnet.py
    fcn.py
experiments/
  your_experiment_info_such_as_logs_and_checkpoints/
```

I provide an example for the [COCO Stuff Dataset](https://github.com/nightrome/cocostuff). One can imitate this example to implement any project. 

## Setup

Install [Anaconda](https://anaconda.org/) and then run
```bash
conda create env
```
