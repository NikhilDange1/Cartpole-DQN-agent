import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQN():

    def __init__(self, no_states , no_action):
        self.states  = no_states
        self.actions = no_action
        self.rlmod = Sequential()
        self.rlmod.add(Dense(units=64, activation='relu', input_dim=self.states))
        self.rlmod.add(Dense(self.actions, activation='linear'))
        self.rlmod.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])


    def memory(self, size):
        self.size = size
        self.mem = []

    def push(self, val):# tuple of state,action,reward,next state,done
        self.mem.append(val)
        if len(self.mem) > self.size:
            del self.mem[0]

    def sample(self,batch_size):
        sample = zip(*random.sample(self.mem,batch_size))
        return sample

    def learn(self,batch,gamma):
        obs, r, a, next_obs, done = self.sample(batch) #random samples from memory
        pred_target = self.rlmod.predict(np.array(obs).reshape(batch, self.states)) #predicted q-values
        next_op1 = self.rlmod.predict(np.array(next_obs).reshape(batch, self.states)) # actual q-values
        for i in range(batch):
            k = a[i]  # k is the index of action of the action taken
            if done[i] == False:
                target = r[i] + gamma * np.amax(next_op1[i][:]) #For non terminal states
            else:
                target = r[i] #For terminal states
            pred_target[i][k] = target
        self.rlmod.fit(np.array(obs).reshape(batch, self.states), pred_target.reshape(batch, self.actions), epochs=1, batch_size=1, verbose=0) #training the network to approximate q-values

    def action_select(self,x, eph):# e-greedy action selection
        q_out = self.rlmod.predict_on_batch(np.array(x).reshape(1, self.states))
        action = np.argmax(q_out)
        act = np.array([0, 1, action])
        return np.random.choice(act, p=[(eph / 2), (eph / 2), (1 - eph)])