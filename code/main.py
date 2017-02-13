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


def sampleAction(observation, iteration_no, iteration_max):
    # TODO: best possible action

    sampled_action = sess.run(output,{observation = observation}

    # TODO: Softmax policy

    # TODO: epsilon-greedy

    # TODO: epsilon as function of time

    # TODO: random
    output = env.observation_space.sample()

    return output


'''
Policy net
'''


def value_gradient(environment_num_observations, environment_num_actions):
    '''
    Creates and runs an inference net that calculates a possible outcome
    :param environment_num_observations: Number of input layer neurons
    :param environment_num_actions: Number of output layer neurons
    :return: output of the net
    '''
    # TODO: tanh vs ReLU vs leaky ReLU
    observation = tf.placeholder("float", [None,environment_num_observations])
    hidden = slim.fully_connected(observation, int(np.mean(observation.get_shape[0], environment_num_actions)),
                                  scope="hidden")
    output = slim.fully_connected(hidden, environment_num_actions, activation_fn=tf.nn.softmax, scope="output")

    newvals = tf.placeholder("float", [None, 1])
    diff = output - newvals
    loss = tf.nn.l2_loss(diff)
    optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)


'''
Initialize plot, environment and TensorFlow
'''
env_name = 'CartPole-v0'
env = gym.make(env_name)

plt.ion()
plt.axes()

sess = tf.Session()
tf.reset_default_graph()
init = tf.initialize_all_variables()
sess.run(init)

state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

MAX_EPISODES = 10000
MAX_STEPS = 200

episode_history = deque(maxlen=100)
start = time.time()
scores = [0]
episode_data = []
for i_episode in range(MAX_EPISODES):
    # initialize
    state = env.reset()
    total_rewards = 0

    for t in range(MAX_STEPS):
        # TODO: Pick action
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        total_rewards += reward
        episode_data.append((state, action, reward))
        state = next_state
        if done:
            # TODO: Calculate discounted rewards, replace immediate rewards with long-term reward
            break

    # TODO: Discount rewards



    # TODO: Calculate gradient
    # TODO: Execute gradient update
    optimizer = tf.train.AdamOptimizer(0.01).compute_gradients()

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
        print("Environment {} solved after {} episodes".format(env_name, i_episode + 1))
        plt.clf()
        plt.plot(scores, color='b')
        if len(scores) % 2 == 0:
            yhat = savgol_filter(scores[0:-1], len(scores) - 1, 4)
        else:
            yhat = savgol_filter(scores, len(scores), 3)
        plt.plot(yhat, linewidth=2, color='r')
        plt.waitforbuttonpress()
        break
