# ESC499 Thesis - Modeling Visuomotor Learning with Adaptive Neural Network Control

This repo contains the code simulating the 3 stages of visuomotor learning (fast, slow, offline) using a recurrent control architecture using adaptive feedforward neural network components to model motor command generation and cerebellar control.

A link to a detailed explanation of the structure will be posted in the near future.

## Dependency Setup

Install **anaconda2** and **Tensorflow** using ```conda install```

## Running

After pulling the repository, the NNs must be first trained. To train the NNs (on a world with no disturbance), go into [simulation.py](simulation.py) and change the first instance of

```python
...
    myRRSim = RRSimulator('./models/RRModel_FM.ckpt', './models/RRModel_IM.ckpt', False, False)
...
```

to 

```python
...
    myRRSim = RRSimulator('./models/RRModel_FM.ckpt', './models/RRModel_IM.ckpt', True, True)
...
```

Change it back after the models are saved to avoid retraining when rerunning the same simulations.

To run the simulation in terminal:

```
python simulation.py
```