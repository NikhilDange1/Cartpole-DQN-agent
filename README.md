# Cartpole-DQN-agent
A DQN agent implemented in python using Keras with TensorFlow as backend. This algorithm uses a neural network as a function approximator
to approximate Q-values and uses an Ephsilon-greedy action selection policy. The state transitions along with reward and action are stored
in a memory. Training the network on a batch of random samples from the memory eliminates the correlation betweeen consecutive states or observations.

## Prerequisite
Python 3.6 is required to run the two files. The code also uses the following additional modules

- [Numpy](http://www.numpy.org)
- [Keras](https://keras.io/#installation)
- [Tensorflow(backend)](https://www.tensorflow.org/install/)
- [OpenAI gym](https://github.com/openai/gym)

## Files
- **DQNagent.py**-This file contains the DQN class which contains the network structure and related methods. The number of states or observations and the number of actions are the arguments needed to create a class instance. The Dqnagent class can be used to solve other reinforcement problems as well and the only changes required would be the network parameters.
- **cartpole.py**-This file runs the simulation and the DQN class. The hyperparameters are in this file(except for the network size, loss function and activation function).
```
agent.memory(100000)
batch = 64 #trainig batch size
eph = 0.9 #ephsilon
eph_min = 0.01
decay = 0.995 #decay for ephsilon
gamma = 0.99 # discount factor
```
