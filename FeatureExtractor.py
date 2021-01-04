import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


class FeatureExtractor:
    def __init__(self, state_dim, hidden_size=64, learning_rate=1e-4, seed=1, max_seq_length=32):
        self.state_dim = state_dim
        self.sess = tf.Session()
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        tf.set_random_seed(seed)
        np.random.seed(seed)
        self.input_state = tf.placeholder(dtype=tf.float32, shape=[None, self.max_seq_length, self.state_dim])
        self.input_state_length = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.input_reward = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.feature = self.create_model_v2(self.input_state, self.input_state_length)
        self.weight = tf.Variable(
            tf.random_normal(shape=[self.hidden_size * 2, 1], stddev=0.3, dtype=tf.float32))  # 这个要注意改动,convLSTM要乘以2
        self.bias = tf.Variable(tf.constant(0.0, dtype=tf.float32))
        self.l2_norm = tf.nn.l2_loss(self.weight) + tf.nn.l2_loss(self.bias)
        expected_output = tf.nn.relu(tf.matmul(self.feature, self.weight) + self.bias)
        self.loss = tf.reduce_mean((expected_output - self.input_reward) ** 2) + 1e-5 * self.l2_norm
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def create_model(self, input_state, input_length):
        '''
        build the rnn model.
        :return: rnn model.
        '''
        with tf.variable_scope('feature_extract', reuse=False):
            basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size)
            _, states = tf.nn.dynamic_rnn(basic_cell, input_state, dtype=tf.float32,
                                          sequence_length=input_length)
        return states[0]

    def create_model_v2(self, input_state, input_length):
        '''
        build the convLSTM model.
        :param input_state: input state
        :param input_length: input length
        :return: the model
        '''
        with tf.variable_scope('feature_extract_v2', reuse=False):
            basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size)
            states, _ = tf.nn.dynamic_rnn(basic_cell, input_state, dtype=tf.float32, sequence_length=input_length)
            features = tf.contrib.slim.conv2d(states, num_outputs=self.hidden_size, kernel_size=3)
            max_pool = tf.reduce_max(features, axis=1)
            mean_pool = tf.reduce_mean(features, axis=1)
            net = tf.concat([max_pool, mean_pool], axis=1)
            net = tf.reshape(net, [-1, 2 * self.hidden_size])
        return net

    def train(self, state, state_length, reward):
        feed_state = np.reshape(state, [-1, self.max_seq_length, self.state_dim])
        feed_length = np.reshape(state_length, [-1, ])
        feed_reward = np.reshape(reward, [-1, ])
        self.sess.run(self.train_op, feed_dict={self.input_state: feed_state, self.input_state_length: feed_length,
                                                self.input_reward: feed_reward})
        loss = self.sess.run(self.loss, feed_dict={self.input_state: feed_state, self.input_state_length: feed_length,
                                                   self.input_reward: feed_reward})
        return loss

    def get_feature(self, state, length):
        state = np.reshape(state, [-1, self.max_seq_length, self.state_dim])
        return self.sess.run(self.feature, feed_dict={self.input_state: state, self.input_state_length: length})

    def get_loss(self, state, state_length, reward):
        feed_state = np.reshape(state, [-1, self.max_seq_length, self.state_dim])
        feed_length = np.reshape(state_length, [-1, ])
        feed_reward = np.reshape(reward, [-1, ])
        loss = self.sess.run(self.loss, feed_dict={self.input_state: feed_state, self.input_state_length: feed_length,
                                                   self.input_reward: feed_reward})
        return loss
