import random
import logging
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim


class DeepQNetworkModel:
    def __init__(self,
                 session,
                 layers_size,
                 memory,
                 tf,
                 default_batch_size=None,
                 default_learning_rate=None,
                 default_epsilon=None,
                 gamma=0.99,
                 min_samples_for_predictions=0,
                 double_dqn=False,
                 learning_procedures_to_q_target_switch=1000,
                 tau=1,
                 maximize_entropy=False,
                 var_scope_name=None,):
        """
        Create a new Deep Q Network model
        :param session: a tf.Session to be used
        :param layers_size: a list of numbers, representing the number of nodes in each layer of the network
        :param memory: an instance of type memory_buffers.Memory
        :param default_batch_size: the default batch size for training
        :param default_learning_rate: the default learning rate for training
        :param default_epsilon: the default epsilon to be used for the eps-greedy policy
        :param gamma: the discount factor
        :param min_samples_for_predictions: the minimum number of seen state-transitions required to make predictions.
                 random numbers will be selected until this number has reached
        :param double_dqn: boolean, should a Double Deep Q Network should be used or not
        :param learning_procedures_to_q_target_switch: how many learning procedures are required before the main network
                 is copied to the q-target network. relevant only if double_dqn = True.
        :param tau: a number in the range [0,1] determining the mixture of the main network weights and q-target weights
                 which will be inserted to q-target. tau=1 copies the main network weights to the q-target network as
                 they are (as should be according to the original paper). tau=0 will keep q-target weights unchanged,
                 meaning no knowledge will be transferred.
                 relevant only if double_dqn = True.
        :param maximize_entropy: boolean, determining if the network should try to optimize the Q values entropy
        :param var_scope_name: when more than one model are generated, each needs its own variable scope. If the two
                 or more models are suppose to share their weights, they both should have the same variable scope name.
                 This is irrelevant when only one instance of the model is used.
        """
        self.output_size = layers_size[-1]
        self.session = session
        self.default_batch_size = default_batch_size
        self.default_learning_rate = default_learning_rate
        self.default_epsilon = default_epsilon
        self.min_samples_for_predictions = min_samples_for_predictions
        self.learning_procedures_to_q_target_switch = learning_procedures_to_q_target_switch
        self.tau = tau
        self.maximize_entropy = maximize_entropy
        self.memory = memory
        self.tf = tf
        self.gamma = gamma
        self.maximize_entropy = maximize_entropy
        self.q_network = self.__create_q_network(input_size=layers_size[0], output_size=self.output_size,
                                                 hidden_layers_size=layers_size[1:-1], gamma=gamma,
                                                 maximize_entropy=maximize_entropy,
                                                 var_scope_name=var_scope_name,
                                                 layer_name_suffix='qnn')
        if double_dqn:
            self.target_q_network = self.__create_q_network(input_size=layers_size[0], output_size=self.output_size,
                                                            hidden_layers_size=layers_size[1:-1], gamma=gamma,
                                                            maximize_entropy=maximize_entropy,
                                                            var_scope_name=var_scope_name,
                                                            layer_name_suffix='qt')
        else:
            self.target_q_network = None
        if not self.tf:
            self.optimizer = optim.Adam(self.q_network.parameters())

    def __create_q_network(self, input_size, output_size, hidden_layers_size, gamma, maximize_entropy,
                           var_scope_name, layer_name_suffix):
        if self.tf:
            scope_name = var_scope_name or tf.get_variable_scope().name
            reuse = tf.AUTO_REUSE if var_scope_name else False
            with tf.variable_scope(scope_name, reuse=reuse):
                qnn = QNetworkTF(input_size=input_size, output_size=output_size, hidden_layers_size=hidden_layers_size,
                                 gamma=gamma, maximize_entropy=maximize_entropy, layer_name_suffix=layer_name_suffix)
        else:
            qnn = QNetwork(input_size=input_size, output_size=output_size, hidden_layers_size=hidden_layers_size)

        return qnn

    def learn(self, learning_rate=None, batch_size=None):
        """
        Initialize a learning attempt
        :param learning_rate: a learning rate overriding default_learning_rate
        :param batch_size: a batch_size overriding default_batch_size
        :return: None if no learning was made, or the cost of learning if it did happen
        """
        if not self.tf:
            criterion = nn.MSELoss()
        current_batch_size = batch_size if batch_size is not None else self.default_batch_size
        if self.memory.counter % current_batch_size != 0 or self.memory.counter == 0:
            logging.debug('Passing on learning procedure')
            pass
        else:
            logging.debug('Starting learning procedure...')
            batch = self.memory.sample(current_batch_size)
            if self.tf:
                qt = self.session.run(self.target_q_network.output,
                                  feed_dict={self.target_q_network.states: self.__fetch_from_batch(batch, 'next_state')})
            else:
                self.target_q_network.eval()
                qt = self.target_q_network(torch.as_tensor(self.__fetch_from_batch(batch, 'next_state'), dtype=torch.float32))

            terminals = self.__fetch_from_batch(batch, 'is_terminal')
            for i in range(terminals.size):
                if terminals[i]:
                    if self.tf:
                        qt[i] = np.zeros(self.output_size)
                    else:
                        qt[i] = torch.zeros(self.output_size)  # manually setting q-target values of terminal states to 0
            lr = learning_rate if learning_rate is not None else self.default_learning_rate
            if self.tf:
                _, cost = self.session.run([self.q_network.optimizer, self.q_network.cost],
                                       feed_dict={self.q_network.states: self.__fetch_from_batch(batch, 'state'),
                                                  self.q_network.r: self.__fetch_from_batch(batch, 'reward'),
                                                  self.q_network.enumerated_actions: self.__fetch_from_batch(batch, 'action', enum=True),
                                                  self.q_network.q_target: qt,
                                                  self.q_network.learning_rate: lr})
            else:
                self.q_network.train()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.optimizer.zero_grad()
                y_hat = self.q_network(torch.as_tensor(self.__fetch_from_batch(batch, 'state'), dtype=torch.float32).requires_grad_())
                if self.maximize_entropy:
                    future_q = torch.logsumexp(qt, dim=1)
                else:
                    future_q = torch.max(qt, dim=1)[0]
                cost = criterion(torch.as_tensor(self.__fetch_from_batch(batch, 'reward'), dtype=torch.float32).requires_grad_() +
                    self.gamma * future_q,
                                 y_hat.gather(1,torch.as_tensor(self.__fetch_from_batch(batch, 'action'), dtype=torch.long).unsqueeze(1)).squeeze())
                cost.backward()
                self.optimizer.step()

            logging.debug('Batch number: %s | Q-Network cost: %s | Learning rate: %s',
                          self.memory.counter // current_batch_size, cost, lr)
            if self.target_q_network is not None and self.memory.counter % (self.learning_procedures_to_q_target_switch * current_batch_size) == 0:
                if (self.tf):
                    logging.info('Copying Q-Network to Q-Target...')
                    tf_vars = tf.trainable_variables()
                    num_of_vars = len(tf_vars)
                    operations = []
                    for i, v in enumerate(tf_vars[0:num_of_vars // 2]):
                        operations.append(tf_vars[i + num_of_vars // 2].assign(
                            (v.value() * self.tau) + ((1 - self.tau) * tf_vars[i + num_of_vars // 2].value())))
                    self.session.run(operations)
                else:
                    self.target_q_network.load_state_dict(self.q_network.state_dict())
            return float(cost)

    def act(self, state, epsilon=None):
        """
        Select an action for the given state
        :param state: a Numpy array representing a state
        :param epsilon: an epsilon value to be used for the eps-greedy policy, overriding default_epsilon
        :return: a number representing the selected action
        """
        eps = epsilon if epsilon is not None else self.default_epsilon
        rnd = random.random()
        if rnd < eps or self.memory.counter < self.min_samples_for_predictions:
            action = random.randint(0, self.output_size - 1)
            logging.debug("Choosing a random action: %s [Epsilon = %s]", action, eps)
        else:
            if self.tf:
                prediction = self.session.run(self.q_network.output, feed_dict={self.q_network.states: np.expand_dims(state, axis=0)})
                prediction = np.squeeze(prediction)
            else:
                self.q_network.eval()
                prediction = self.q_network(torch.as_tensor(np.expand_dims(state, axis=0), dtype=torch.float32)).detach().numpy().squeeze()
            action = np.argmax(prediction)
            logging.debug("Predicted action for state %s is %s (network output: %s) [Epsilon = %s]",
                          state, action, prediction, eps)
        return action

    def add_to_memory(self, state, action, reward, next_state, is_terminal_state):
        """
        Add new state-transition to memory
        :param state: a Numpy array representing a state
        :param action: an integer representing the selected action
        :param reward: a number representing the received reward
        :param next_state: a Numpy array representing the state reached after performing the action
        :param is_terminal_state: boolean. mark state as a terminal_state. next_state will have no effect.
        """
        self.memory.append({'state': state, 'action': action, 'reward': reward,
                            'next_state': next_state, 'is_terminal': is_terminal_state})

    def __fetch_from_batch(self, batch, key, enum=False):
        if enum:
            return np.array(list(enumerate(map(lambda x: x[key], batch))))
        else:
            return np.array(list(map(lambda x: x[key], batch)))

    def save_network(self, filename):
        torch.save(self.q_network, filename)

    def load_network(self, filename):
        self.q_network = torch.load(filename, map_location=torch.device('cpu'))


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
    def __init__(self, input_size, output_size, hidden_layers_size, gamma, maximize_entropy, layer_name_suffix):
        self.q_target = tf.placeholder(shape=(None, output_size), dtype=tf.float32)
        self.r = tf.placeholder(shape=None, dtype=tf.float32)
        self.states = tf.placeholder(shape=(None, input_size), dtype=tf.float32)
        self.enumerated_actions = tf.placeholder(shape=(None, 2), dtype=tf.int32)
        self.learning_rate = tf.placeholder(shape=[], dtype=tf.float32)
        layer = self.states
        for i in range(len(hidden_layers_size)):
            layer = tf.layers.dense(inputs=layer, units=hidden_layers_size[i], activation=tf.nn.relu,
                                    name='{}_dense_layer_{}'.format(layer_name_suffix,i),
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.output = tf.layers.dense(inputs=layer, units=output_size,
                                      name='{}_dense_layer_{}'.format(layer_name_suffix,len(hidden_layers_size)),
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.predictions = tf.gather_nd(self.output, indices=self.enumerated_actions)
        if maximize_entropy:
            self.future_q = tf.log(tf.reduce_sum(tf.exp(self.q_target), axis=1))
        else:
            self.future_q = tf.reduce_max(self.q_target, axis=1)
        self.labels = self.r + (gamma * self.future_q)
        self.cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.labels, predictions=self.predictions))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
