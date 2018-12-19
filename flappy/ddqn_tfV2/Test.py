import tensorflow as tf
from ddqn_tfV2.Game import Game
import numpy as np
from ddqn_tfV2.Hyperparameters import *
# tf.reset_default_graph()
saver = tf.train.import_meta_graph(model_path+"-3400.meta")

with tf.Session() as sess:
    # Load the models
    graph = tf.get_default_graph()
    saver.restore(sess, tf.train.latest_checkpoint(tensor_dir))
    inputs_ = tf.placeholder(tf.float32, [None, state_size], name="inputs")
    # w1 =  sess.run('DQNetwork/w1:0')
    # b1 =  sess.run('DQNetwork/b1:0')
    #
    # w2 = sess.run('DQNetwork/w2:0')
    # b2 = sess.run('DQNetwork/b2:0')
    #
    # w3 = sess.run('DQNetwork/w3:0')
    # b3 = sess.run('DQNetwork/b3:0')
    #
    # l1 = tf.nn.relu(tf.matmul(inputs_, w1) + b1)
    # l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
    # Q = tf.matmul(l2, w3) + b3
    inputs_ = graph.get_tensor_by_name("DQNetwork" + "/inputs_:0")
    Q = graph.get_tensor_by_name("DQNetwork" + "/Q:0")
    render = True
    game = Game(render)
    for i in range(100):
        # start new game
        done = False
        action = 0
        stack_frames=np.reshape(game.stack_frames,[1,state_size])
        while done != True:
            # state = np.float32(state)
            q_state = sess.run(Q, feed_dict={inputs_: stack_frames})
            action = np.argmax(q_state)
            (stack_frames, action, reward, next_state, done) = game.play(action, i)
            stack_frames=[next_state]
        print(game.total_score)
        game.reset(render,False)
