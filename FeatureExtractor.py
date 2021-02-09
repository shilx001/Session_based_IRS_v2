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
        self.input_state_length = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.input_reward = tf.placeholder(dtype=tf.float32, shape=[None, ])
        # self.feature = self.create_model_v5(self.input_state, self.input_state_length)
        self.feature = self.create_model(self.input_state, self.input_state_length)
        # self.feature = self.create_model_atem(self.input_state)
        #self.feature = self.create_model_caser(self.input_state, self.input_state_length)
        #self.feature = self.create_model_v2(self.input_state, self.input_state_length)
        #self.feature = self.create_model_v3(self.input_state, self.input_state_length)
        self.weight = tf.Variable(
            tf.random_normal(shape=[self.hidden_size, 1], stddev=0.3, dtype=tf.float32))  # 这个要注意改动,convLSTM要乘以2
        self.bias = tf.Variable(tf.constant(0.0, dtype=tf.float32))
        self.l2_norm = tf.nn.l2_loss(self.weight) + tf.nn.l2_loss(self.bias)
        expected_output = tf.matmul(self.feature, self.weight) + self.bias
        self.value_loss = tf.reduce_mean((expected_output - self.input_reward) ** 2)
        self.loss = self.value_loss + 1e-5 * self.l2_norm
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
            features = slim.conv2d(states, self.hidden_size, kernel_size=3)
            max_pool = tf.reduce_max(features, axis=1)
            mean_pool = tf.reduce_mean(features, axis=1)
            net = tf.concat([max_pool, mean_pool], axis=1)
            net = tf.reshape(net, [-1, 2 * self.hidden_size])
        return net

    def create_model_v3(self, input_state, input_length):
        with tf.variable_scope('feature_extract_v3', reuse=False):
            basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size)
            item_embedding, _ = tf.nn.dynamic_rnn(basic_cell, input_state, dtype=tf.float32, sequence_length=input_length)
            attention_layer = slim.fully_connected(item_embedding, 1)
            attention_weight = tf.nn.softmax(attention_layer, 1)  # [N, max_seq_length, 1]
            output_layer = attention_weight * item_embedding  # [N, 1, hidden_size]
            output_layer = tf.reduce_sum(output_layer, axis=1)
        return output_layer

    def create_model_v4(self, input_state, input_length):
        '''
        build the convLSTM model.
        :param input_state: input state
        :param input_length: input length
        :return: the model
        '''
        with tf.variable_scope('feature_extract_v2', reuse=False):
            basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size)
            states, _ = tf.nn.dynamic_rnn(basic_cell, input_state, dtype=tf.float32, sequence_length=input_length)
            features = slim.conv2d(states, num_outputs=self.hidden_size, kernel_size=5)
            max_pool = tf.reduce_max(features, axis=1)
        return max_pool

    def create_model_caser(self, input_state, input_length):
        '''
        build the Caser model.
        :param input_state:
        :param input_length:
        :return:
        '''
        with tf.variable_scope('feature_extract_caser', reuse=False):
            input_state = tf.expand_dims(input_state, axis=3)
            output_v = tf.layers.conv2d(input_state, self.hidden_size, [self.max_seq_length, 1],
                                        activation=tf.nn.relu)
            out_v = tf.layers.flatten(output_v)  # [N, self.state_dim*self.hidden_size]
            output_hs = list()
            for h in np.arange(self.max_seq_length) + 1:
                conv_out = tf.layers.conv2d(input_state, self.hidden_size, [h, self.state_dim],
                                            activation=tf.nn.relu)
                conv_out = tf.reshape(conv_out, [-1, self.max_seq_length - h + 1, self.hidden_size])
                pool_out = tf.layers.max_pooling1d(conv_out, [self.max_seq_length + 1 - h], 1)
                pool_out = tf.squeeze(pool_out, 1)
                output_hs.append(pool_out)
            out_h = tf.concat(output_hs, axis=1)
            out = tf.concat([out_v, out_h], axis=1)
            z = tf.layers.dense(out, self.hidden_size, activation=tf.nn.relu)
        return z

    def create_model_atem(self, input_state):
        '''
        build the ATEM model
        :param input_state: input state.
        :param input_length: input state length
        :return: the ATEM model
        '''
        with tf.variable_scope('feature_extractor_atem', reuse=False):
            item_embedding = slim.fully_connected(input_state, self.hidden_size)  # [N, max_seq_length,hidden_size]
            attention_layer = slim.fully_connected(item_embedding, 1)
            attention_weight = tf.nn.softmax(attention_layer, 1)  # [N, max_seq_length, 1]
            output_layer = attention_weight * item_embedding  # [N, 1, hidden_size]
            output_layer = tf.reduce_sum(output_layer, axis=1)
        return output_layer

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
        loss = self.sess.run(self.value_loss, feed_dict={self.input_state: feed_state, self.input_state_length: feed_length,
                                                   self.input_reward: feed_reward})
        return loss

    def state_padding(self, input_state, input_state_length):
        if input_state_length > self.max_seq_length:
            input_state = input_state[-self.max_seq_length:]
            input_state_length = self.max_seq_length
        input_state = np.array(input_state).reshape([input_state_length, self.state_dim])
        if input_state_length < self.max_seq_length:
            # padding the zero matrix.
            padding_mat = np.zeros([self.max_seq_length - input_state_length, self.state_dim])
            input_state = np.vstack((input_state, padding_mat))
        return input_state
