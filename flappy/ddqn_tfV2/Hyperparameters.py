### MODEL HYPERPARAMETERS
state_size = 16  # Our input is a stack of 4 frames hence 100x120x4 (Width, height, channels)
action_size = 2  # 7 possible actions
learning_rate = 0.001  # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 5000  # Total episodes for training
batch_size = 64

# FIXED Q TARGETS HYPERPARAMETERS
max_tau = 1000  # Tau is the C step where we update our target network

# EXPLORATION HYPERPARAMETERS for epsilon greedy strategy
explore_start = 1.0  # exploration probability at start
explore_stop = 0.001  # minimum exploration probability
decay_rate = 0.000005  # exponential decay rate for exploration prob

# Q LEARNING hyperparameters
gamma = 0.95 # Discounting rate

### MEMORY HYPERPARAMETERS
## If you have GPU change to 1million
pretrain_length = 10000  # Number of experiences stored in the Memory when initialized for the first time
memory_size = 10000  # Number of experiences the Memory can keep

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

#hidden layer for network
no_hidden_layer=24

# restore the model weights
restore_weights=False
# tensor_dir="./tensorboard/dddqn/1"
# model_path="./models/models.ckpt"

import os
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

model_name="model_tf_13"
score_plot_path=script_dir+"/plots/"+model_name+".png"
tensor_dir=script_dir+"/tensorboard/"+model_name
model_path=script_dir+"/models/"+model_name+".ckpt"

render=False
isPrioritized=True

f = open(script_dir+"/models/"+model_name+".txt", "a")
for name, value in globals().copy().items():
    param=name+" "+ str(value)+"\n"
    f.write(param)
f.close()