import datetime
import time
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy.signal import savgol_filter

'''
Exploration function
'''
def sampleAction(iteration_no, iteration_max):
    #TODO: epsilon-greedy
    #TODO: epsilon as function of time
    #TODO: random
    return env.observation_space.sample()

'''
Policy net
'''
def inference(observation):
    #TODO: tanh vs ReLU vs leaky ReLU
    hidden = slim.fully_connected(observation, 3, scope="hidden")
    output = slim.fully_connected(hidden, 2, activation_fn=tf.nn.softmax, scope="output")
    return output


'''
Initialize plot, environment and TensorFlow
'''
env_name = 'CartPole-v0'
env = gym.make(env_name)

plt.ion()
plt.axes()

sess = tf.Session()

state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

MAX_EPISODES = 10000
MAX_STEPS = 200

episode_history = deque(maxlen=100)
start = time.time()
scores = [0]
for i_episode in range(MAX_EPISODES):
    # initialize
    state = env.reset()
    total_rewards = 0

    for t in range(MAX_STEPS):
        action = sampleAction(i_episode, MAX_EPISODES)  # This is where the magic happens
        next_state, reward, done, _ = env.step(action)

        total_rewards += reward
        reward = -10 if done else 0.1  # normalize reward TODO: <??
        pg_reinforce.storeRollout(state, action, reward)

        state = next_state
        if done: break

    #TODO: Calculate gradient
    #TODO: Execute gradient update

    episode_history.append(total_rewards)
    mean_rewards = np.mean(episode_history)
    scores.append(mean_rewards)
    if i_episode % 100 == 1:
        plt.clf()
        plt.plot(scores, color='b')
        if len(scores) > 3:
          yhat = savgol_filter(scores, len(scores), 4)
          plt.plot(yhat, linewidth=2, color='r')
        plt.pause(1e-30)
        print("Episode {}".format(i_episode))
        print("Time elapsed: {}".format(datetime.timedelta(seconds=time.time() - start)))
        print("Average reward for last 100 episodes: {}".format(mean_rewards))
    if mean_rewards >= 195.0 and len(episode_history) >= 100:
        print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
        plt.clf()
        plt.plot(scores, color='b')
        if len(scores) % 2 == 0:
            yhat = savgol_filter(scores[0:-1], len(scores)-1, 4)
        else:
            yhat = savgol_filter(scores, len(scores), 3)
        plt.plot(yhat, linewidth=2, color='r')
        plt.waitforbuttonpress()
        break
