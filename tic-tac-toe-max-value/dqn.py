import random
import tensorflow as tf
from collections import deque
import torch.nn as nn

class QNetwork(nn.Module):
    """
    A Q-Network implementation
    input: vector of 9 cells - current state of the game
    output: vector of 9 values for each action possible to input state
    """
    def __init__(self, input_size, output_size, hidden_layers_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_layers_size[0])
        nn.init.xavier_uniform_(self.fc1.weight)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layers_size[0], hidden_layers_size[1])
        nn.init.xavier_uniform_(self.fc2.weight)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_layers_size[1], hidden_layers_size[2])
        nn.init.xavier_uniform_(self.fc3.weight)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_layers_size[2], hidden_layers_size[3])
        nn.init.xavier_uniform_(self.fc4.weight)
        self.relu4 = nn.ReLU()
        self.out = nn.Linear(hidden_layers_size[3], output_size)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        a2 = self.fc2(h1)
        h2 = self.relu2(a2)
        a3 = self.fc3(h2)
        h3 = self.relu3(a3)
        a4 = self.fc4(h3)
        h4 = self.relu4(a4)
        y = self.out(h4)
        return y

class QNetworkTF:
    """
    A Q-Network implementation
    """
    def __init__(self, input_size, output_size, hidden_layers_size, gamma):
        self.q_target = tf.placeholder(shape=(None, output_size), dtype=tf.float32)
        self.r = tf.placeholder(shape=None, dtype=tf.float32)
        self.states = tf.placeholder(shape=(None, input_size), dtype=tf.float32)
        self.actions = tf.placeholder(shape=(None, 2), dtype=tf.int32)  # enumerated actions
        self.learning_rate = tf.placeholder(shape=[], dtype=tf.float32)
        layer = self.states
        for l in hidden_layers_size:
            layer = tf.layers.dense(inputs=layer, units=l, activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.output = tf.layers.dense(inputs=layer, units=output_size,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.predictions = tf.gather_nd(self.output, indices=self.actions)
        self.labels = self.r + (gamma * tf.reduce_max(self.q_target, axis=1))
        self.cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.labels, predictions=self.predictions))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

class ReplayMemory:
    """
    A cyclic Experience Replay memory buffer
    """
    memory: deque = None
    counter = 0

    def __init__(self, size, seed=None):
        self.memory = deque(maxlen=size)
        if seed is not None:
            random.seed(seed)

    def __len__(self):
        return len(self.memory)

    def append(self, element):
        self.memory.append(element)
        self.counter += 1

    def sample(self, n, or_less=False):
        if or_less and n > self.counter:
            n = self.counter
        return random.sample(self.memory, n)
