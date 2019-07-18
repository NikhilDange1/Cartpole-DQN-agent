# Cartpole-DQN-agent
A Deep Reinforcement Learningn agent implemented in python using Keras with TensorFlow as backend. This algorithm can be found in the [Playing Atari with Deep Reinforcement Learning paper](https://arxiv.org/pdf/1312.5602.pdf)


## Prerequisite
The code runs on Python 3.6 and uses the following modules

- [Numpy](http://www.numpy.org)
- [Keras](https://keras.io/#installation)
- [Tensorflow(backend)](https://www.tensorflow.org/install/)
- [OpenAI gym](https://github.com/openai/gym)

Modules can be installed using the requirements.txt file

## Files
- **DQNagent.py**-The file contains the Deep Q netwrok agent object.

- **Env.py**-This file runs the OpenAI gym simulation  and calls the DQN class.

## Hyperparameters
```
agent.memory(100000)
batch = 64 #trainig batch size
eph = 0.9 #ephsilon
eph_min = 0.01
decay = 0.995 #decay for ephsilon
gamma = 0.99 # discount factor
```
