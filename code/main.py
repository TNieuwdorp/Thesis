from __future__ import print_function

import datetime
import time
import matplotlib.pyplot as plt
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from agent import PolicyGradientREINFORCE
from scipy.signal import savgol_filter


'''
Initialize plot, environment and TensorFlow
'''
env_name = 'CartPole-v0'
env = gym.make(env_name)

plt.ion()
plt.axes()

sess = tf.Session()
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
writer = tf.summary.FileWriter("/tmp/{}-experiment-1".format(env_name))

state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

def policy_network(states):
  # define policy neural network
  W1 = tf.get_variable("W1", [state_dim, 20],
                       initializer=tf.random_normal_initializer())
  b1 = tf.get_variable("b1", [20],
                       initializer=tf.constant_initializer(0))
  h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
  W2 = tf.get_variable("W2", [20, num_actions],
                       initializer=tf.random_normal_initializer(stddev=0.1))
  b2 = tf.get_variable("b2", [num_actions],
                       initializer=tf.constant_initializer(0))
  p = tf.matmul(h1, W2) + b2
  return p

pg_reinforce = PolicyGradientREINFORCE(sess,
                                       optimizer,
                                       policy_network,
                                       state_dim,
                                       num_actions,
                                       summary_writer=writer)

MAX_EPISODES = 10000
MAX_STEPS    = 200

episode_history = deque(maxlen=100)
start = time.time()
scores = [0]
for i_episode in range(MAX_EPISODES):

  # initialize
  state = env.reset()
  total_rewards = 0

  for t in range(MAX_STEPS):
    # env.render()
    action = pg_reinforce.sampleAction(state[np.newaxis,:])
    next_state, reward, done, _ = env.step(action)

    total_rewards += reward
    reward = -10 if done else 0.1 # normalize reward
    pg_reinforce.storeRollout(state, action, reward)

    state = next_state
    if done: break

  pg_reinforce.updateModel()

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
