import gym
import numpy as np
from DQNagent import DQN




env = gym.make('CartPole-v0')

actions = env.action_space.n
states = env.observation_space.shape[0]
agent = DQN(states, actions)

total_episodes = 1000
batch = 64
eph = 0.9 #ephsilon
eph_min = 0.01
decay = 0.995 #decay for ephsilon

count =0
average_rewards=[]

for episodes in range(total_episodes):
    current_state = env.reset()
    total_reward = 0
    while True:
        env.render()
        if count == 0:
            action = env.action_space.sample()
        else:
            action = agent.action_select(current_state,eph)
        next_state , reward , done , _  = env.step(action)
        total_reward+=reward
        agent.push([current_state, reward, action, next_state, done]) #push to memory
        current_state = next_state
        if count > agent.batch_size:
            agent.learn()
        count+=1
        if eph > eph_min:
            eph*=decay
        if done:
            average_rewards.append(total_reward)
            break
    if len(average_rewards) > 100:
        del average_rewards[0] #moving average
    if np.mean(average_rewards)> 195:
        break
    print("Episode: ",episodes,"Reward: ",total_reward,"Average: ", np.mean(average_rewards))
print('solved after ',episodes,' episodes')