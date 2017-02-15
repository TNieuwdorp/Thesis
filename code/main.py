import gym
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

tf.reset_default_graph()

# Parameters
env = gym.make('CartPole-v0')
gamma = 0.99
lr = 0.1
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
hidden_size = 10
max_episodes = -1  # Set total number of episodes to train agent on.
max_steps = 1000
batch_size = 15
strategy = 3
# 1: Argmax
# 2: Softmax
# 3: Epsilon-greedy
# 4: Decaying epsilon-greedy
# else: Random


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def leaky_relu(x, alpha=0.01):
    """Leaky ReLU."""
    return tf.maximum(alpha * x, x)


'''
# Define the agent
'''
# These lines established the feed-forward part of the network. The agent takes a state and produces an action.
state_in = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
hidden = slim.fully_connected(state_in, hidden_size, activation_fn=leaky_relu)
output = slim.fully_connected(hidden, output_size, activation_fn=tf.nn.softmax,
                              biases_initializer=None)


# The next six lines establish the training procedure. We feed the reward and chosen action into the network
# to compute the loss, and use it to update the network.
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

init = tf.global_variables_initializer()

# Launch the TensorFlow graph
sess = tf.Session()
sess.run(init)
i = 0
total_reward = []
total_length = []
timer = time.time()
show_result = False

# Initialize gradient buffer with zero
gradBuffer = sess.run(tvars)
for ix, grad in enumerate(gradBuffer):
    gradBuffer[ix] = 0

while i != max_episodes:
    if time.time() - timer > 5:
        timer = time.time()
        show_result = True
    s = env.reset()
    running_reward = 0
    ep_history = []
    for step in range(max_steps):
        # Choose either a random action or one from our network.
        a_dist = sess.run(output, feed_dict={state_in: [s]})
        a = np.random.choice(a_dist[0], p=a_dist[0])
        a = np.argmax(a_dist == a)
        '''
        if strategy == 1:
            # Argmax
            a = tf.argmax(output, 1)
        elif strategy == 2:
            # Softmax policy
            a = np.random.choice(env.action_space, p=output)
        elif strategy == 3:
            # epsilon-greedy
            if np.random.uniform(0, 1) < 0.8:
                a = tf.argmax(output, 1)
            else:
                a = env.action_space.sample()
        elif strategy == 4:
            # decaying epsilon-greedy
            if np.random.uniform(0, 1) < 1:
                a = tf.argmax(output, 1)
            else:
                a = env.action_space.sample()
        else:
            # Random
            a = env.observation_space.sample()
        '''
        s1, r, d, _ = env.step(a)  # Get our reward for taking an action given a bandit.
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

            if i % batch_size == 0 and i != 0:
                # Average buffer over batch size
                gradBuffer[:] = [x / batch_size for x in gradBuffer]
                feed_dict = dict(zip(gradient_holders, gradBuffer))
                _ = sess.run(update_batch, feed_dict=feed_dict)
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

            total_reward.append(running_reward)
            break
        if show_result:
            env.render()

    # Update our running tally of scores.
    if show_result:
        print("Iteration " + str(i) + ": " + str(np.mean(total_reward[-100:])))
        show_result = False
    i += 1
