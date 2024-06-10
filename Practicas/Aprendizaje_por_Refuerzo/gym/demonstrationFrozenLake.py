# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:32:39 2022

Demonstration of the OpenAI Gym Library 
and the Fronzen Lake reinforcement learning environment

@author: Aleksandar Haber 

Website accompanying this code with background information
and theoretical explanations is given here:

https://aleksandarhaber.com/introduction-to-state-transition-probabilities-actions-and-rewards-with-openai-gym-reinforcement-learning-tutorial/

"""
import gymnasium as gym

env=gym.make("FrozenLake-v1", render_mode="human", is_slippery=True)
# env = gym.make('FrozenLake-v1', render_mode='human', map_name="8x8", is_slippery=True)

env.reset()
# render the environment
env.render()

# observation space - states 
print(env.observation_space)

# actions: left -0, down - 1, right - 2, up- 3
print(env.action_space)

#generate random action
randomAction= env.action_space.sample()
print(randomAction)

returnValue = env.step(randomAction)
print(returnValue)

# format of returnValue is (observation,reward, terminated, truncated, info)
# observation (object)  - observed state
# reward (float)        - reward that is the result of taking the action
# terminated (bool)     - is it a terminal state
# truncated (bool)      - it is not important in our case
# info (dictionary)     - in our case transition probability


env.render()

# perform deterministic step 0,1,2,3
returnValue = env.step(1)

# reset the environment
env.reset()

#transition probabilities
#p(s'|s,a) probability of going to state s' 
#          starting from the state s and by applying the action a

# env.P[state][action]
print(env.unwrapped.P[9][0] )
print(env.unwrapped.P[9][1] )
print(env.unwrapped.P[9][2] )
print(env.unwrapped.P[9][3] )
# output is a list having the following entries
# (transition probability, next state, reward, Is terminal state?)

# close the environment
env.close()
