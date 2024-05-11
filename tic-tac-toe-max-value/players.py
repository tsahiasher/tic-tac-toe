import logging
import random
import dqn
import numpy as np
usingTensorFlow = False
if usingTensorFlow:
    import tensorflow as tf
else:
    import torch
    import torch.nn as nn
    import torch.optim as optim
from abc import abstractmethod

class Player:
    """
    Base class for all player types
    """
    name = None
    player_id = None

    def __init__(self):
        pass

    def shutdown(self):
        pass

    def add_to_memory(self, add_this):
        pass

    @abstractmethod
    def select_cell(self, board, **kwargs):
        pass

    @abstractmethod
    def learn(self, **kwargs):
        pass


class Human(Player):
    """
    This player type allow a human player to play the game
    """
    def select_cell(self, board, **kwargs):
        while True:
            try:
                cell = int(input("Select cell to fill:\n678\n345\n012\ncell number: "))
                if cell < 0 or cell > 8:
                    raise ValueError
            except ValueError:
                print("Oops! That was no valid number. Try again...")
            else:
                break

        return cell

    def learn(self, **kwargs):
        pass


class Drunk(Player):
    """
    Drunk player always selects a random valid move
    """
    def select_cell(self, board, **kwargs):
        available_cells = np.where(board == 0)[0]
        return random.choice(available_cells)

    def learn(self, **kwargs):
        pass


class Novice(Player):
    """
    A more sophisticated bot, which follows the following strategy:
    1) If it already has 2-in-a-row, capture the required cell for 3
    2) If not, and if the opponent has 2-in-a-row, capture the required cell to prevent hi, from winning
    3) Else, select a random vacant cell
    """
    def find_two_of_three(self, board, which_player_id):
        cell = None
        winning_options = [[0, 1, 2], [3, 4, 5], [6, 7, 8],
                           [0, 3, 6], [1, 4, 7], [2, 5, 8],
                           [0, 4, 8], [2, 4, 6]]
        random.shuffle(winning_options)
        for seq in winning_options:
            s = board[seq[0]] + board[seq[1]] + board[seq[2]]
            if s == 2 * which_player_id:
                a = np.array([board[seq[0]], board[seq[1]], board[seq[2]]])
                c = np.where(a == 0)[0][0]
                cell = seq[c]
                break
        return cell

    def select_cell(self, board, **kwargs):
        cell = self.find_two_of_three(board,self.player_id)
        if cell is None:
            cell = self.find_two_of_three(board,-self.player_id)
        if cell is None:
            available_cells = np.where(board == 0)[0]
            cell = random.choice(available_cells)
        return cell

    def learn(self, **kwargs):
        pass


class QPlayer(Player):
    """
    A reinforcement learning agent, based on Double Deep Q Network model
    This class holds two Q-Networks: `qnn` is the learning network, `q_target` is the semi-constant network
    """
    def __init__(self, hidden_layers_size, gamma, learning_batch_size,
                 batches_to_q_target_switch, tau, memory_size):
        """
        :param hidden_layers_size: an array of integers, specifying the number of layers of the network and their size
        :param gamma: the Q-Learning discount factor
        :param learning_batch_size: training batch size
        :param batches_to_q_target_switch: after how many batches (trainings) should the Q-network be copied to Q-Target
        :param tau: a number between 0 and 1, determining how to combine the network and Q-Target when copying is performed
        :param memory_size: size of the memory buffer used to keep the training set
        """
        self.learning_batch_size = learning_batch_size
        self.batches_to_q_target_switch = batches_to_q_target_switch
        self.tau = tau
        self.learn_counter = 0
        self.counter = 0
        self.memory = dqn.ReplayMemory(memory_size)
        if usingTensorFlow:
            self.session = tf.Session()
            self.qnn = dqn.QNetworkTF(9, 9, hidden_layers_size, gamma)
            self.q_target = dqn.QNetworkTF(9, 9, hidden_layers_size, gamma)
            self.session.run(tf.global_variables_initializer())
        else:
            self.qnn = dqn.QNetwork(9, 9, hidden_layers_size)
            self.gamma = gamma
            self.q_target = dqn.QNetwork(9, 9, hidden_layers_size)
            self.optimizer = optim.Adam(self.qnn.parameters())
        super(QPlayer, self).__init__()

    def select_cell(self, board, **kwargs):
        rnd = random.random()
        eps = kwargs['epsilon']
        self.counter += 1
        if rnd < eps:
            cell = random.randint(0,8)
            logging.debug("Choosing a random cell: %s [Epsilon = %s]", cell, eps)
        else:
            if usingTensorFlow:
                prediction = self.session.run(self.qnn.output,feed_dict={self.qnn.states: np.expand_dims(self.player_id * board, axis=0)})
                prediction = np.squeeze(prediction)
            else:
                self.qnn.eval()
                prediction = self.qnn(torch.as_tensor(np.expand_dims(self.player_id * board, axis=0), dtype=torch.float32)).detach().numpy().squeeze()
            cell = np.argmax(prediction)
            logging.debug("Predicting next cell - board: %s | player ID: %s | prediction: %s | cell: %s [Epsilon = %s]", board, prediction, cell, eps)
        return cell

    @staticmethod
    def _fetch_from_batch(batch, key, enum=False):
        if enum:
            return np.array(list(enumerate(map(lambda x: x[key], batch))))
        else:
            return np.array(list(map(lambda x: x[key], batch)))

    def learn(self, **kwargs):
        logging.debug('Memory counter = %s', self.memory.counter)
        self.learn_counter += 1
        if not usingTensorFlow:
            criterion = nn.MSELoss()

        if self.learn_counter % self.learning_batch_size != 0 or self.memory.counter < self.learning_batch_size:
            pass
        else:
            logging.debug('Starting learning procedure')
            batch = self.memory.sample(self.learning_batch_size)
            if usingTensorFlow:
                qt = self.session.run(self.q_target.output,feed_dict={self.q_target.states: self._fetch_from_batch(batch,'next_state')})
            else:
                self.q_target.eval()
                qt = self.q_target(torch.as_tensor(self._fetch_from_batch(batch,'next_state'), dtype=torch.float32))
            terminals = self._fetch_from_batch(batch,'game_over')
            for i in range(terminals.size):
                if terminals[i]:
                    if usingTensorFlow:
                        qt[i] = np.zeros(9)  # manually setting q-target values of terminal states to 0
                    else:
                        qt[i] = torch.zeros(9)  # manually setting q-target values of terminal states to 0
            lr = kwargs['learning_rate']
            if usingTensorFlow:
                _, cost = self.session.run([self.qnn.optimizer, self.qnn.cost],
                                           feed_dict={self.qnn.states: self._fetch_from_batch(batch, 'state'),
                                                      self.qnn.r: self._fetch_from_batch(batch, 'reward'),
                                                      self.qnn.actions: self._fetch_from_batch(batch, 'action', enum=True),
                                                      self.qnn.q_target: qt,
                                                      self.qnn.learning_rate: lr})
            else:
                self.qnn.train()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                self.optimizer.zero_grad()
                y_hat = self.qnn(torch.as_tensor(self._fetch_from_batch(batch, 'state'), dtype=torch.float32).requires_grad_()) #get value for each action per state in batch
                cost = criterion(torch.as_tensor(self._fetch_from_batch(batch, 'reward'), dtype=torch.float32).requires_grad_() +
                                 self.gamma * torch.max(qt, dim=1)[0],
                                 y_hat.gather(1, torch.as_tensor(self._fetch_from_batch(batch, 'action'), dtype=torch.long).unsqueeze(1)).squeeze())
                cost.backward()
                self.optimizer.step()

            logging.debug('Batch number: %s | Q-Network cost: %s | Learning rate: %s',
                         self.learn_counter // self.learning_batch_size, float(cost), lr)

            if self.learn_counter % (self.batches_to_q_target_switch * self.learning_batch_size) == 0:
                logging.info('Copying Q-Network to Q-Target')
                if usingTensorFlow:
                    tf_vars = tf.trainable_variables()
                    num_of_vars = len(tf_vars)
                    operations = []
                    for i,v in enumerate(tf_vars[0:num_of_vars//2]):
                        operations.append(tf_vars[i+num_of_vars//2].assign((v.value()*self.tau) + ((1-self.tau)*tf_vars[i+num_of_vars//2].value())))
                    self.session.run(operations)
                else:
                    self.q_target.load_state_dict(self.qnn.state_dict())

            return float(cost)

    def add_to_memory(self, add_this):
        add_this['state'] = self.player_id * add_this['state']
        add_this['next_state'] = self.player_id * add_this['next_state']
        self.memory.append(add_this)

    def save(self, filename):
        if usingTensorFlow:
            saver = tf.train.Saver()
            saver.save(self.session, filename)
        else:
            torch.save(self.qnn, filename)

    def restore(self, filename):
        if usingTensorFlow:
            saver = tf.train.Saver()
            saver.restore(self.session, filename)
        else:
            self.qnn = torch.load(filename, map_location=torch.device('cpu'))

    def shutdown(self):
        if usingTensorFlow:
            self.session.close()
        else:
            return


