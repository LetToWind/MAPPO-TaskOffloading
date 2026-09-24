# MADDPG_Discrete

This folder contains an experimental discrete-action MADDPG variant.

It keeps the original MEC environment dynamics and reward calculation from
`MADDPG/Environment_marl_3.py`, but replaces the actor action representation:

- Original baseline: two continuous actor outputs are scaled and cast with `int()`.
- Discrete version: each agent outputs decomposed categorical decisions:
  - each channel chooses `idle` or one local task slot;
  - each task chooses one power level.

During training the actor uses straight-through Gumbel-Softmax so gradients can
flow through the discrete heads. The environment adapter encodes the decomposed
choices back into the original strategy IDs before calling the original `step()`
logic, keeping the experimental environment comparable with the paper baseline.

Run from this folder:

```bash
python maddpg_discrete.py
```

Useful environment variables:

```bash
N_EPISODES=20000
BATCH_SIZE=256
MEMORY_SIZE=50000
```
