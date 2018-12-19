import tensorflow as tf      # Deep Learning library
from ddqn_tfV2.Hyperparameters import *



import warnings # This ignore all
class DDDQNNet:
    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        # We use tf.variable_scope here to know which network we're using (DQN or target_net)
        # it will be useful when we will update our w- parameters (by copy the DQN parameters)
        with tf.variable_scope(self.name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote

            w_initializer=tf.initializers.random_uniform()
            b_initializer=tf.constant_initializer(0.1)

            self.inputs_ = tf.placeholder(tf.float32, [None, self.state_size], name="inputs")
            if(isPrioritized):self.ISWeights_ = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
            self.actions_ = tf.placeholder(tf.float32, [None,self.action_size], name="actions_")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            # self.target_Q2 = tf.placeholder(tf.float32, [None, self.action_size], name="target2")

            w1 = tf.get_variable('w1', [self.state_size, no_hidden_layer], initializer=w_initializer,trainable=True)
            b1 = tf.get_variable('b1', [1, no_hidden_layer], initializer=b_initializer, trainable=True)
            l1 = tf.nn.relu(tf.matmul(self.inputs_, w1) + b1)

            w2 = tf.get_variable('w2', [no_hidden_layer, no_hidden_layer], initializer=w_initializer, trainable=True)
            b2 = tf.get_variable('b2', [1, no_hidden_layer], initializer=b_initializer, trainable=True)
            l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            w3 = tf.get_variable('w3', [no_hidden_layer, self.action_size], initializer=w_initializer,trainable=True)
            b3 = tf.get_variable('b3', [1, self.action_size], initializer=b_initializer,trainable=True)

            # Q is our predicted Q value.
            self.Q = tf.matmul(l2, w3) + b3

            # index = np.arange(64, dtype=np.int32)

            self.Q_ = tf.reduce_sum(tf.multiply(self.Q, self.actions_), axis=1)


            # The loss is modified because of PER
            self.absolute_errors = tf.abs(self.target_Q - self.Q_)  # for updating Sumtree
            # self.t1=tf.squared_difference(self.target_Q, self.Q_);
            # self.t2= tf.squared_difference(self.target_Q2, self.Q);
            if (isPrioritized):
                self.loss = tf.reduce_mean(self.ISWeights_ *tf.squared_difference(self.target_Q, self.Q_))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.target_Q, self.Q_))
            # self.loss2 = tf.reduce_mean(self.ISWeights_ * tf.reduce_sum(tf.squared_difference(self.target_Q2, self.Q), axis=1))

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            # self.optimizer2 = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss2)