import tensorflow as tf
from ddqn_tfV2.Game import Game
import numpy as np
from ddqn_tfV2.Hyperparameters import *
tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph(script_dir+"/models/model_tf_12.ckpt.meta")

with tf.Session() as sess:
    # Load the models
    imported_meta.restore(sess, tf.train.latest_checkpoint(script_dir+'/models/'))
    inputs_ = tf.placeholder(tf.float32, [None, state_size], name="inputs")
    w1 =  sess.run('DQNetwork/w1:0')
    b1 =  sess.run('DQNetwork/b1:0')

    w2 = sess.run('DQNetwork/w2:0')
    b2 = sess.run('DQNetwork/b2:0')

    w3 = sess.run('DQNetwork/w3:0')
    b3 = sess.run('DQNetwork/b3:0')

    l1 = tf.nn.relu(tf.matmul(inputs_, w1) + b1)
    l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
    Q = tf.matmul(l2, w3) + b3

    # playerpos = [100, 100]
    # width, height = 640, 480
    # state_size = 5
    # state = np.reshape([playerpos[0], playerpos[1], height / 2, height / 2, height / 2],
    #                         [1, state_size])
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
        game.reset(render)
