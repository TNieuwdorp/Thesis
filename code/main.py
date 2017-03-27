import time
from enum import Enum
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
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


def leaky_relu(x, alpha=5.5):
    """Leaky ReLU."""
    return tf.maximum(x / alpha, x)


# Parameters
env = gym.make('CartPole-v0')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

report_every_s = 10

max_episodes = 5000  # Set total number of episodes to train agent on.
max_steps = 500

epsilon = 0.4
epsilon_decay = .9995
gamma = 0.99
lr = 0.01
hidden_size = 10
batch_size = 1
# strategy = Strategy.E_GREEDY

'''
# Define the policy network
'''
# These lines established the feed-forward part of the network. The agent takes a state and produces an action.
state_in = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
hidden = slim.fully_connected(state_in, hidden_size, activation_fn=leaky_relu)
output = slim.fully_connected(hidden, output_size, activation_fn=tf.nn.softmax,
                              biases_initializer=None)

'''
# Training procedure. Compute the loss based on chosen action and reward and update the net accordingly.
'''
reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

indexes = tf.range(0, tf.shape(output)[0]) * tf.shape(output)[1] + action_holder
responsible_outputs = tf.gather(tf.reshape(output, [-1]), indexes)

loss = -tf.reduce_mean(tf.log(responsible_outputs) * reward_holder)

tvars = tf.trainable_variables()
gradient_holders = []
for idx, var in enumerate(tvars):
    placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
    gradient_holders.append(placeholder)

gradients = tf.gradients(loss, tvars)

optimizer = tf.train.AdamOptimizer(learning_rate=lr)
update_batch = optimizer.apply_gradients(zip(gradient_holders, tvars))

# Initialize TensorFlow graph
init = tf.global_variables_initializer()

# Initialize variables for capturing results
num_trials = 30
results = np.zeros((num_trials, len(Strategy), 3))

for strategy in Strategy:
    for trial in range(0, num_trials):
        # Launch the TensorFlow graph
        with tf.Session() as sess:
            sess.run(init)
            start_time = time.time()
            timer = time.time()
            show_result = True
            # Initialize gradient buffer with zero
            gradBuffer = sess.run(tvars)
            for ix, grad in enumerate(gradBuffer):
                gradBuffer[ix] = 0
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

                    s1, r, d, _ = env.step(a)
                    ep_history.append([s, a, r, s1])
                    s = s1
                    running_reward += r
                    if d:
                        # Update the network.
                        ep_history = np.array(ep_history)
                        ep_history[:, 2] = discount_rewards(ep_history[:, 2])
                        feed_dict = {reward_holder: ep_history[:, 2],
                                     action_holder: ep_history[:, 1], state_in: np.vstack(ep_history[:, 0])}
                        grads = sess.run(gradients, feed_dict=feed_dict)
                        for idx, grad in enumerate(grads):
                            gradBuffer[idx] += grad

                        # Collect a batch of episodes and get an average gradient out of these
                        if i % batch_size == 0 and i != 0:
                            # Average buffer over batch size
                            gradBuffer[:] = [x / batch_size for x in gradBuffer]
                            feed_dict = dict(zip(gradient_holders, gradBuffer))
                            _ = sess.run(update_batch, feed_dict=feed_dict)
                            for ix, grad in enumerate(gradBuffer):
                                gradBuffer[ix] = grad * 0

                        total_reward.append(running_reward)
                        break

                # Update our running tally of scores.
                if show_result:
                    print("Strategy: " + str(strategy.name) + "; Trial: " + str(trial) + "; Iteration " + str(i) + ": "
                          + str(np.mean(total_reward[-100:])))
                    show_result = False

                results[trial, strategy.value, :] = [i, time.time() - start_time, np.mean(total_reward[-100:])]
                # Print when task is completed
                if np.mean(total_reward[-100:]) > 195 and i > 100:
                    print("--- Task completed after: " + str(i) + " iterations in " + str(
                        int(time.time() - start_time)) + " seconds. ---")
                    break

# Plot the results
iterations = results[:, :, 0]
time = results[:, :, 1]
score = results[:, :, 2]

iterations_df = pd.DataFrame(iterations)
ax = sns.boxplot(iterations_df)
ax.set(xlabel='Strategy', ylabel='No of iterations')
ax.set_xticklabels([x.name for x in Strategy])
ax.set_ylim(0,)
sns.plt.show()

time_df = pd.DataFrame(time)
ax = sns.boxplot(time_df)
ax.set(xlabel='Strategy', ylabel='No of seconds')
ax.set_xticklabels([x.name for x in Strategy])
ax.set_ylim(0,)
sns.plt.show()

score_df = pd.DataFrame(score)
ax = sns.boxplot(score_df)
ax.set(xlabel='Strategy', ylabel='Score')
ax.set_xticklabels([x.name for x in Strategy])
ax.set_ylim(0,)
sns.plt.show()
