import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

class TreePolicy:
    def __init__(self, state_dim, layer=3, branch=32, hidden_size=64, learning_rate=1e-4, seed=1,
                 stddev=0.03):
        self.state_dim = state_dim
        self.sess = tf.Session()
        self.layer = layer
        self.branch = branch
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.stddev = stddev
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        self.input_state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        self.input_action = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.input_reward = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.tree = self.create_tree()
        self.output_action_prob = self.forward_pass()
        action_mask = tf.one_hot(self.input_action, self.branch ** self.layer)  # output the action of each node.
        prob_under_policy = tf.reduce_sum(self.output_action_prob * action_mask, axis=1)
        self.loss = -tf.reduce_mean(self.input_reward * tf.log(prob_under_policy + 1e-13), axis=0)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def mlp(self, id=None):
        '''
        Create a multi-layer neural network as tree node.
        :param id: tree node id
        :param reuse: reuse for the networks
        :return: a multi-layer neural network with output dim equals to branch size.
        '''
        with tf.variable_scope('node_' + str(id), reuse=False):
            l1 = slim.fully_connected(self.input_state, self.hidden_size)
            l2 = slim.fully_connected(l1, self.hidden_size)
            l3 = slim.fully_connected(l2, self.branch)
            outputs = tf.nn.softmax(l3)
        return outputs  # [N, branch]

    def create_tree(self):
        '''
        Build the tree-structure policy.
        :return: a list of nodes, each item denotes a layer.
        '''
        # total_nodes = int((self.branch ** self.layer - 1) / (self.branch - 1))
        layer_nodes = []
        for i in range(self.layer):
            current_layer = [self.mlp(id=str(i) + '_' + str(_)) for _ in range(int(self.branch ** i))]
            layer_nodes.append(current_layer)
        return layer_nodes

    def forward_pass(self):
        '''
        Calculate output probability for each item.
        :return: a tensor of the tree policy.
        '''
        root_node = self.tree[0]
        root_output = root_node[0]
        for i in range(1, self.layer):  # for each layer
            current_output = []
            for j in range(self.branch ** i):  # for each leaf node
                current_layer = self.tree[i]
                current_output.append(tf.expand_dims(root_output[:, j], axis=1) * current_layer[j])
            root_output = tf.concat(current_output, axis=1)  # [N, branch**i], update root_output.
        return root_output

    def get_action_prob(self, state):
        '''
        get probability for each action.
        :param state: input state, shape=[N, state_dim].
        :return: the probability for each action.
        '''
        return self.sess.run(self.output_action_prob, feed_dict={self.input_state: state})

    def train(self, state, action, reward):
        '''
        Update the gradient of the policy network.
        :param state: input state.
        :param action: input action.
        :param reward: input return.
        :return: the loss value of each update.
        '''
        state = np.reshape(state, [-1, self.state_dim])
        action = np.reshape(action, [-1, ])
        reward = np.reshape(reward, [-1, ])
        loss = self.sess.run(self.loss, feed_dict={self.input_state: state, self.input_action: action,
                                                   self.input_reward: reward})
        self.sess.run(self.train_step, feed_dict={self.input_state: state, self.input_action: action,
                                                  self.input_reward: reward})
        return loss
