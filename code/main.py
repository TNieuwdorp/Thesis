import os
import time
from enum import Enum
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import seaborn as sns
import pandas as pd

import gym


class Strategy(Enum):
    RANDOM = 0
    ARGMAX = 1
    SOFTMAX = 2
    E_GREEDY = 3
    DECAY_E_GREEDY = 4


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# Parameters
env = gym.make('CartPole-v0')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

report_every_s = 5

max_episodes = 1000  # Set total number of episodes to train agent on.
max_steps = 500

epsilon = 0.3
epsilon_decay = .99
gamma = 0.99
lr = 0.05
strategy = Strategy.E_GREEDY

'''
# Define the policy network
'''
# These lines established the feed-forward part of the network. The agent takes a state and produces an action.
state_in = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
output = slim.fully_connected(state_in, output_size,
                              activation_fn=tf.nn.softmax,
                              biases_initializer=None)

'''
# Training procedure.
'''
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
action_holder = tf.placeholder(dtype=tf.int32)  # One-hot encoded vector
return_holder = tf.placeholder(dtype=tf.float32)
baseline = tf.constant(value=0, dtype=tf.float32)  # For improved variance this can be set to average expected return
#  for the given state

# The network's predicted probability of the performed action
network_prediction = tf.reduce_sum(tf.multiply(output, tf.one_hot(action_holder, env.action_space.n)), axis=1)
log_probabilities = tf.log(network_prediction)
# We need to maximize the function below, so minimize it's negative (optimizer only supports minimize)
loss = -tf.reduce_sum(lr * tf.multiply(log_probabilities, (return_holder - baseline)))
# Function below calculates the gradients of trainable variable given the above loss function as function to minimize
optimize = optimizer.minimize(loss)  # Requires state_in = s, action_holder = a, reward_holder = r

# Create an operation to initialize TensorFlow graph and variables
init = tf.global_variables_initializer()

# Initialize variables for capturing results
num_trials = 30
results = np.zeros((num_trials, len(Strategy), 3))

for strategy in Strategy:
    for trial in range(0, num_trials):
        # Launch the TensorFlow graph
        with tf.Session() as sess:
            sess.run(init)  # Initialize graph and variables
            start_time = time.time()
            timer = start_time
            show_result = True
            total_reward = []

            for i in range(max_episodes):
                # Report status every report_every_s seconds
                if time.time() - timer > report_every_s:
                    timer = time.time()
                    show_result = True

                s = env.reset()
                running_reward = 0
                ep_history = []
                for step in range(max_steps):
                    # Choose an action (depends on exploration strategy)
                    net_a_dist = sess.run(output, feed_dict={state_in: [s]}).flatten()

                    if strategy == Strategy.ARGMAX:
                        a = np.argmax(net_a_dist)
                    elif strategy == Strategy.SOFTMAX:
                        # Softmax policy
                        a = np.random.choice([0, 1], p=net_a_dist)
                    elif strategy == Strategy.E_GREEDY:
                        # epsilon-greedy
                        if np.random.uniform(0, 1) < epsilon:
                            a = env.action_space.sample()
                        else:
                            a = np.argmax(net_a_dist)
                    elif strategy == Strategy.DECAY_E_GREEDY:
                        # decaying epsilon-greedy
                        if np.random.uniform(0, 1) < np.power(epsilon_decay, i):
                            a = env.action_space.sample()
                        else:
                            a = np.argmax(net_a_dist)
                    else:
                        # Random
                        a = env.action_space.sample()

                    s_new, r, d, _ = env.step(a)
                    ep_history.append([s, a, r, s_new])
                    s = s_new
                    running_reward += r
                    if d:
                        # Update the network using discounted reward as Q-value
                        ep_history = np.array(ep_history)
                        ep_history[:, 2] = discount_rewards(ep_history[:, 2])

                        feed_dict = {return_holder: ep_history[:, 2],
                                     action_holder: ep_history[:, 1],
                                     state_in: np.vstack(ep_history[:, 0])}
                        grads = sess.run(optimize, feed_dict=feed_dict)
                        total_reward.append(running_reward)
                        break

                # Update our running tally of scores.
                if show_result:
                    print("Strategy: " + str(strategy.name) + "; Trial: " + str(trial) + "; Iteration " + str(i) +
                          ": " + str(np.mean(total_reward[-100:])))
                    show_result = False

                results[trial, strategy.value, :] = [i, time.time() - start_time, np.mean(total_reward[-100:])]
                # Print when task is completed
                if np.mean(total_reward[-100:]) > 195 and i > 100:
                    print("--- Task completed after: " + str(i) + " iterations in " + str(int(time.time() - start_time))
                          + " seconds. ---")
                    break

# Plot the results
iterations = results[:, :, 0]
time = results[:, :, 1]
score = results[:, :, 2]

iterations_df = pd.DataFrame(iterations)
ax = sns.boxplot(iterations_df)
ax.set(xlabel='Strategy', ylabel='No of iterations')
ax.set_xticklabels(x.name for x in Strategy)
ax.set_ylim(0,)
sns.plt.show()

time_df = pd.DataFrame(time)
ax = sns.boxplot(time_df)
ax.set(xlabel='Strategy', ylabel='No of seconds')
ax.set_xticklabels(x.name for x in Strategy)
ax.set_ylim(0,)
sns.plt.show()

score_df = pd.DataFrame(score)
ax = sns.boxplot(score_df)
ax.set(xlabel='Strategy', ylabel='Score')
ax.set_xticklabels(x.name for x in Strategy)
ax.set_ylim(0,)
sns.plt.show()
