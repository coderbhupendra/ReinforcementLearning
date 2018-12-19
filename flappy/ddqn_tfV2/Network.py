import tensorflow as tf      # Deep Learning library
from ddqn_tfV2.Hyperparameters import *



import warnings # This ignore all
class DDDQNNet:
    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        if restore:
            # for op in tf.get_default_graph().get_operations():
            #     print(str(op.name))

            graph = tf.get_default_graph()
            # w1 = graph.get_tensor_by_name(name+"/w1:0")
            # b1 = graph.get_tensor_by_name(name+"/b1:0")
            # w2 = graph.get_tensor_by_name(name + "/w2:0")
            # b2 = graph.get_tensor_by_name(name + "/b2:0")
            # w3 = graph.get_tensor_by_name(name + "/w3:0")
            # b3 = graph.get_tensor_by_name(name + "/b3:0")
            self.inputs_ = graph.get_tensor_by_name(name+"/inputs_:0")
            self.actions_ = graph.get_tensor_by_name(name+"/actions_:0")
            self.target_Q = graph.get_tensor_by_name(name+"/target_Q:0")
            if (isPrioritized): self.ISWeights_ = graph.get_tensor_by_name(name+"/ISWeights_:0")

            self.Q=graph.get_tensor_by_name(name + "/Q:0")
            self.Q_ = graph.get_tensor_by_name(name + "/Q_:0")
            self.absolute_errors = graph.get_tensor_by_name(name + "/absolute_errors:0")
            self.loss = graph.get_tensor_by_name(name + "/loss:0")
            self.optimizer = graph.get_operation_by_name(name + "/optimizer")


        # We use tf.variable_scope here to know which network we're using (DQN or target_net)
        # it will be useful when we will update our w- parameters (by copy the DQN parameters)
        else:
            with tf.variable_scope(self.name):
                # We create the placeholders
                # *state_size means that we take each elements of state_size in tuple hence is like if we wrote

                w_initializer=tf.initializer=tf.contrib.layers.variance_scaling_initializer()
                b_initializer=tf.initializer=tf.contrib.layers.variance_scaling_initializer()

                self.inputs_ = tf.placeholder(tf.float32, [None, self.state_size], name="inputs_")
                if(isPrioritized):self.ISWeights_ = tf.placeholder(tf.float32, [None, 1], name='ISWeights_')
                self.actions_ = tf.placeholder(tf.float32, [None,self.action_size], name="actions_")
                # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
                self.target_Q = tf.placeholder(tf.float32, [None], name="target_Q")
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
                self.Q = tf.add(tf.matmul(l2, w3) , b3,name="Q")

                # index = np.arange(64, dtype=np.int32)

                self.Q_ = tf.reduce_sum(tf.multiply(self.Q, self.actions_), axis=1,name="Q_")


                # The loss is modified because of PER
                self.absolute_errors = tf.abs(self.target_Q - self.Q_,name="absolute_errors")  # for updating Sumtree
                # self.t1=tf.squared_difference(self.target_Q, self.Q_);
                # self.t2= tf.squared_difference(self.target_Q2, self.Q);
                if (isPrioritized):
                    self.loss = tf.reduce_mean(self.ISWeights_ *tf.squared_difference(self.target_Q, self.Q_),name="loss")
                else:
                    self.loss = tf.reduce_mean(tf.squared_difference(self.target_Q, self.Q_),name="loss")
                # self.loss2 = tf.reduce_mean(self.ISWeights_ * tf.reduce_sum(tf.squared_difference(self.target_Q2, self.Q), axis=1))

                self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss,name="optimizer")
                # self.optimizer2 = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss2)