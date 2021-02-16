import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


class SharedTreeActorCritic:
    def __init__(self, state_dim, layer=3, branch=32, hidden_size=64, a_lr=1e-4, c_lr=1e-3, seed=1,
                 stddev=0.03, discount_factor=0.99, max_seq_length=32):
        self.state_dim = state_dim
        self.sess = tf.Session()
        self.layer = layer
        self.branch = branch
        self.hidden_size = hidden_size
        self.seed = seed
        self.stddev = stddev
        self.discount_factor = discount_factor
        self.max_seq_length = max_seq_length
        self.a_lr, self.c_lr = a_lr, c_lr
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        self.input_state = tf.placeholder(dtype=tf.float32, shape=[None, self.max_seq_length, state_dim])
        self.input_state_length = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.input_next_state = tf.placeholder(dtype=tf.float32, shape=[None, self.max_seq_length, state_dim])
        self.input_next_state_length = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.input_action = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.input_reward = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.done = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.advantage = self.input_reward + self.discount_factor * self.create_critic(
            self.input_next_state, self.input_next_state_length) * (1 - self.done) - \
                         self.create_critic(self.input_state, self.input_state_length)
        self.output_action_prob = self.forward_pass_v3()
        action_mask = tf.one_hot(self.input_action, self.branch ** self.layer)  # output the action of each node.
        prob_under_policy = tf.reduce_sum(self.output_action_prob * action_mask, axis=1)
        self.actor_loss = -tf.reduce_mean(tf.stop_gradient(self.advantage) * tf.log(prob_under_policy + 1e-13), axis=0)
        self.critic_loss = tf.reduce_mean(self.advantage ** 2)
        self.actor_train_step = tf.train.AdamOptimizer(self.a_lr).minimize(self.actor_loss)
        self.critic_train_step = tf.train.AdamOptimizer(self.c_lr).minimize(self.critic_loss)
        self.sess.run(tf.global_variables_initializer())

    def feature_extract(self, input_state, input_state_length):
        '''
        Create RNN feature extractor for recommendation systems.
        :return:
        '''
        with tf.variable_scope('feature_extract', reuse=False):
            basic_cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)
            _, states = tf.nn.dynamic_rnn(basic_cell, input_state, dtype=tf.float32,
                                          sequence_length=input_state_length)
        return states

    def feature_extract_caser(self, input_state):
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

    def feature_extract_atem(self, input_state):
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

    def mlp(self, id=None, softmax_activation=False):
        '''
        Create a multi-layer neural network as tree node for the actor network.
        :param id: tree node id
        :param reuse: reuse for the networks
        :return: a multi-layer neural network with output dim equals to branch size.
        '''
        with tf.variable_scope('node_' + str(id), reuse=tf.AUTO_REUSE):
            state = self.feature_extract(self.input_state, self.input_state_length)
            l1 = slim.fully_connected(state, self.hidden_size)
            l2 = slim.fully_connected(l1, self.hidden_size)
            l3 = slim.fully_connected(l2, self.hidden_size)
            if softmax_activation:
                outputs = tf.nn.softmax(l3)
            else:
                outputs = l3
        return outputs  # [N, branch]

    def create_critic(self, state, state_length):
        '''
        Create critic network
        :param state: input state
        :param state_length: input state length
        :return: a multi-layer neural network
        '''
        with tf.variable_scope('Critic', reuse=tf.AUTO_REUSE):
            output_state = self.feature_extract(state, state_length)
            l1 = slim.fully_connected(output_state, self.hidden_size)
            l2 = slim.fully_connected(l1, self.hidden_size)
            l3 = slim.fully_connected(l2, 1, activation_fn=None)
        return tf.squeeze(l3)

    def forward_pass_v2(self):
        '''
        Calculate output probability for each item, with shared parameter for each layer.
        :return: a tensor of the tree policy.
        '''
        node = [self.mlp(id=str(_)) for _ in range(self.layer)]
        root_output = node[0]
        for i in range(1, self.layer):  # for each layer
            current_output = []
            for j in range(self.branch ** i):  # for each leaf node
                current_output.append(tf.expand_dims(root_output[:, j], axis=1) * node[i])
            root_output = tf.concat(current_output, axis=1)  # [N, branch**i], update root_output.
        return root_output

    def forward_pass_v3(self):
        '''
        Partial shared layer parameter.
        :return: a tensor of the tree policy.
        '''
        node = [self.mlp(id=str(_), softmax_activation=False) for _ in range(self.layer)]
        root_output = node[0]
        for i in range(1, self.layer):  # for each layer
            current_output = []
            for j in range(self.branch ** i):  # for each leaf node
                current_node = slim.fully_connected(node[i], num_outputs=self.branch, activation_fn=tf.nn.relu)
                current_node = slim.fully_connected(current_node, num_outputs=self.branch, activation_fn=tf.nn.softmax)
                current_output.append(tf.expand_dims(root_output[:, j], axis=1) * current_node)
            root_output = tf.concat(current_output, axis=1)  # [N, branch**i], update root_output.
        return root_output

    def forward_pass_v4(self):
        '''
        Calculate output probability for each item. shared policy with
        :return: a tensor of the tree policy.
        '''
        node = self.mlp(id='node', softmax_activation=False)
        root_output = node
        for i in range(1, self.layer):  # for each layer
            current_output = []
            for j in range(self.branch ** i):  # for each leaf node
                current_node = slim.fully_connected(node, num_outputs=self.branch, activation_fn=tf.nn.relu)
                current_node = slim.fully_connected(current_node, num_outputs=self.branch, activation_fn=tf.nn.softmax)
                current_output.append(tf.expand_dims(root_output[:, j], axis=1) * current_node)
            root_output = tf.concat(current_output, axis=1)  # [N, branch**i], update root_output.
        return root_output

    def get_action_prob(self, state, state_length):
        '''
        get probability for each action.
        :param state: input state, shape=[N, self.max_seq_length, state_dim].
        :param state_length: input state length
        :return: the probability for each action.
        '''
        state = np.reshape(state, [-1, self.max_seq_length, self.state_dim])
        return self.sess.run(self.output_action_prob,
                             feed_dict={self.input_state: state, self.input_state_length: state_length})

    def train(self, state, state_length, action, reward, next_state, next_state_length, done):
        '''
        Update the gradient of the policy network.
        :param state: input state.
        :param state_length: input state length
        :param action: input action.
        :param reward: input return.
        :param next_state: input next state.
        :param next_state_length: input next state length
        :param done: if episodes end then 1 else 0.
        :return: the loss value of each update.
        '''
        state = np.reshape(state, [-1, self.max_seq_length, self.state_dim])
        action = np.reshape(action, [-1, ])
        reward = np.reshape(reward, [-1, ])
        next_state = np.reshape(next_state, [-1, self.max_seq_length, self.state_dim])
        done = np.reshape(done, [-1, ])
        actor_loss, critic_loss, _, _ = self.sess.run([self.actor_loss, self.critic_loss, self.actor_train_step,
                                                       self.critic_train_step],
                                                      feed_dict={self.input_state: state,
                                                                 self.input_state_length: state_length,
                                                                 self.input_action: action,
                                                                 self.input_reward: reward,
                                                                 self.input_next_state: next_state,
                                                                 self.input_next_state_length: next_state_length,
                                                                 self.done: done})
        return actor_loss, critic_loss

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
