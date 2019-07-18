import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQN():

    def __init__(self, no_states , no_action,memory_size=100000,model_layers=[64],batch_size=64,gamma=0.99):
        '''

        :param no_states: The number of states or observation space. This is simply the number of differnet observations and is
                          the size of the input layer of the neural network
        :param no_action: The number of possible actions the agent can take. The agent does not know the actual value of the action
                        but will return the index for that value.
        :param memory_size: The memory size of action replay. The agent stores previous state,rewards,action, current state and done
                            in the memory and randomly samples it
        :param model_layers: A list of int containing the number of neurons in each hidden layer starting from 1st to last. The Finished
                            network will have 2 more layers for input and ouput.
        :param batch_size: integer Batch size for neural network
        :param gamma: float[0,1]Discount factor for the agent
        '''
        self.states  = no_states
        self.actions = no_action
        self.size = memory_size
        self.mem = []
        self.batch_size=batch_size
        self.gamma=gamma
        self.rlmod = self.create_model(model_layers)
        self.rlmod.compile(optimizer=Adam(lr=0.001),loss='mse',metrics=['mae','mse'])
        self.rlmod.summary()


    def create_model(self,model_layers):
        '''

        :param model_layers: List of integers containing neurons in each hidden layer starting from 1st
        :return: Keras model object
        '''
        model=Sequential()
        model.add(Dense(units=model_layers[0], activation='relu', input_dim=self.states))
        if len(model_layers)>1:
            for i in model_layers[1:]:
                model.add(Dense(units=i, activation='relu'))

        model.add(Dense(self.actions, activation='linear'))
        return model


    def push(self, val):
        '''tuple of state,reward,action,,next state,done is pushed onto the memory'''
        self.mem.append(val)
        if len(self.mem) > self.size:
            del self.mem[0]

    def sample(self,batch_size):
        '''
        function to randomly sample state,reward,action,next state,done from memory.
        :param batch_size: integer Number of samples to be taken at once
        :return: (State,reward,action,next state done)
        '''
        sample = zip(*random.sample(self.mem,batch_size))
        return sample

    def learn(self):
        '''
        Function for training the neural network. The function will update the weights of the newtork and does not return anyting
        '''
        obs, r, a, next_obs, done = self.sample(self.batch_size) #random samples from memory
        pred_target = self.rlmod.predict(np.array(obs).reshape(self.batch_size, self.states)) #predicted q-values
        next_op1 = self.rlmod.predict(np.array(next_obs).reshape(self.batch_size, self.states)) # actual q-values
        for i in range(self.batch_size):
            k = a[i]  # k is the index of action of the action taken
            if done[i] == False:
                target = r[i] + self.gamma * np.amax(next_op1[i][:]) #For non terminal states
            else:
                target = r[i] #For terminal states
            pred_target[i][k] = target
        self.rlmod.fit(np.array(obs).reshape(self.batch_size, self.states), pred_target.reshape(self.batch_size, self.actions), epochs=1, batch_size=1, verbose=0) #training the network to approximate q-values

    def action_select(self,x, eph):
        '''
        A function to select an action based on the Epsilon greedy policy. Epislon percent of times the agent will select a random
        action while 1-Epsilon percent of the time the agent will select the action with the highest Q value as predicted by the
        neural network.
        :param x: list of shape [nuber of observation,]
        :param eph: Float [0,1) value of epsilon
        :return: Integer indicating the index of actual action
        '''
        q_out = self.rlmod.predict_on_batch(np.array(x).reshape(1, self.states))
        action = np.argmax(q_out)
        act = np.array([0, 1, action])
        return np.random.choice(act, p=[(eph / 2), (eph / 2), (1 - eph)])