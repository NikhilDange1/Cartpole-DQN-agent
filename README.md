# Cartpole-DQN-agent
A DQN agent implemented in python using Keras with TensorFlow as backend. This algorithm uses a neural network as a function approximator
to approximate Q-values and uses an Ephsilon-greedy action selection policy. The state transitions along with reward and action are stored
in a memory. Training the network on a batch of random samples from the memory eliminates the correlation betweeen consecutive states or observations.
