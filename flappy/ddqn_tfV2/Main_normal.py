import tensorflow as tf
from ddqn_tfV2.Network import DDDQNNet
from ddqn_tfV2.Game import Game
from ddqn_tfV2.Hyperparameters import *
from collections import deque
import numpy as np
import random

# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DDDQNNet(state_size, action_size, learning_rate, name="DQNetwork")

# Instantiate the target network
TargetNetwork = DDDQNNet(state_size, action_size, learning_rate, name="TargetNetwork")

# Instantiate memory
memory = deque(maxlen=5000)
possible_actions = np.identity(2, dtype=int).tolist()

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
game = Game(False)
i=0
while i <pretrain_length:
    #start new game
    done=False
    action=0
    while done!=True:
        action = np.random.randint(0, action_size)
        state, action, reward, next_state, done=game.play(action)
        # print(i,state, action, reward, next_state, done)
        memory.append((state, action, reward, next_state, done))
        i+=1
    game.reset(False)
print("memory Initialsed")

# Setup TensorBoard Writer
writer = tf.summary.FileWriter(tensor_dir)

## Losses
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()

"""
This function will do the part
With ϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
"""


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = np.random.randint(0, action_size)

    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.Q, feed_dict={DQNetwork.inputs_: state})

        # Take the biggest Q value (= the best action)
        action = np.argmax(Qs)

    return action, explore_probability


# This function helps us to copy one set of variables to another
# In our case we use it when we want to copy the parameters of DQN to Target_network
# Thanks of the very good implementation of Arthur Juliani https://github.com/awjuliani
def update_target_graph():
    # Get the parameters of our DQNNetwork
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")

    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

    op_holder = []

    # Update our target_network parameters with DQNNetwork parameters
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Saver will help us to save our models
saver = tf.train.Saver()

if training == True:
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0

        # Set tau = 0
        tau = 0

        # Init the game
        game.reset(render)
        # render = True
        # game = Game(render)
        # Update the parameters of our TargetNetwork with DQN_weights
        update_target = update_target_graph()
        sess.run(update_target)

        for episode in range(total_episodes):
            # Initialize the rewards of the episode
            episode_rewards = []

            done = False
            action = 0
            while done != True:
                # Increase the C step
                tau += 1

                # Increase decay_step
                decay_step += 1

                # With ϵ select a random action atat, otherwise select a = argmaxQ(st,a)
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, [state] )

                # Do the action
                (state, action, reward, next_state, done) = game.play(action, episode)
                # Add experience to memory
                experience = state, action, reward, next_state, done,
                memory.append((experience))

                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the game is finished
                if done:
                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_probability))
                    game.reset(render)



                ### LEARNING PART
                # Obtain random mini-batch from memory
                mini_batch = random.sample(memory, batch_size)

                states_mb,actions_mb,rewards_mb,next_states_mb,dones_mb=[],[],[],[],[]
                for i in range(batch_size):
                    states_mb.append(mini_batch[i][0])
                    actions_mb.append(possible_actions[mini_batch[i][1]])
                    rewards_mb.append(mini_batch[i][2])
                    next_states_mb.append(mini_batch[i][3])
                    dones_mb.append(mini_batch[i][4])

                target_Qs_batch = []

                ### DOUBLE DQN Logic
                # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
                # Use TargetNetwork to calculate the Q_val of Q(s',a')

                # Get Q values for next_state
                q_next_state = sess.run(DQNetwork.Q, feed_dict={DQNetwork.inputs_: next_states_mb})

                # q_state_target = sess.run(DQNetwork.Q, feed_dict={DQNetwork.inputs_: states_mb})
                # Calculate Qtarget for all actions that state
                q_target_next_state = sess.run(TargetNetwork.Q, feed_dict={TargetNetwork.inputs_: next_states_mb})

                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma * Qtarget(s',a')
                for i in range(0, len(states_mb)):
                    terminal = dones_mb[i]

                    # We got a'
                    action = np.argmax(q_next_state[i])

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target=rewards_mb[i]
                        target_Qs_batch.append(target)

                    else:
                        # Take the Qtarget for action a'
                        target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                        target_Qs_batch.append(target)
                    # q_state_target[i][actions[i]]=target
                targets_mb = np.array([each for each in target_Qs_batch])
                _, loss,summary = sess.run([DQNetwork.optimizer,DQNetwork.loss,write_op],
                                                    feed_dict={DQNetwork.inputs_: states_mb,
                                                               # DQNetwork.target_Q2: q_state_target,
                                                               DQNetwork.target_Q: targets_mb,
                                                               DQNetwork.actions_: actions_mb})

                writer.add_summary(summary, episode)
                writer.flush()

                if tau > max_tau:
                    # Update the parameters of our TargetNetwork with DQN_weights
                    update_target = update_target_graph()
                    sess.run(update_target)
                    tau = 0
                    print("Model updated")

            # Save models every 50 episodes
            if episode % 50 == 0:
                save_path = saver.save(sess, model_path)
                print("Model Saved")