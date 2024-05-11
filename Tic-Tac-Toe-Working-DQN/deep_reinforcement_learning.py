import time
#import tensorflow as tf
import random
import numpy as np

import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

boardSize = 9
actions = 9
won_games = 0
lost_games = 0
draw_games = 0
layer_1_w = 750
layer_2_w = 750
layer_3_w = 750

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

# Discount factor -- determines the importance of future rewards
GAMMA = 0.9

# swaps X's to O's and vice versa
def InverseBoard(board):
    return np.negative(board)

def game_status(board):
    winner = 0
    winning_seq = []
    winning_options = [[0, 1, 2], [3, 4, 5], [6, 7, 8],
                       [0, 3, 6], [1, 4, 7], [2, 5, 8],
                       [0, 4, 8], [2, 4, 6]]
    for seq in winning_options:
        s = board[seq[0]] + board[seq[1]] + board[seq[2]]
        if abs(s) == 3:
            winner = s / 3
            winning_seq = seq
            break
    game_over = winner != 0 or len(list(filter(lambda z: z == 0, board))) == 0
    return game_over, winner, winning_seq

def isGameOver(board):
    game_over, _, _ = game_status(board)
    return game_over

class TorchNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(boardSize, layer_1_w)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(layer_1_w, layer_2_w)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(layer_2_w, layer_3_w)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(layer_3_w, actions)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        a2 = self.fc2(h1)
        h2 = self.relu2(a2)
        a3 = self.fc3(h2)
        h3 = self.relu3(a3)
        y = self.out(h3)
        return y

# creates the network
def createNetwork():
    # network weights and biases

    W_layer1 = weight_variable([boardSize, layer_1_w])
    b_layer1 = bias_variable([layer_1_w])

    W_layer2 = weight_variable([layer_1_w, layer_2_w])
    b_layer2 = bias_variable([layer_2_w])

    W_layer3 = weight_variable([layer_2_w, layer_3_w])
    b_layer3 = bias_variable([layer_3_w])

    o_layer = weight_variable([layer_3_w, actions])
    o_bais = bias_variable([actions])

    # input Layer
    x = tf.placeholder("float", [None, boardSize])

    # hidden layers
    h_layer1 = tf.nn.relu(tf.matmul(x, W_layer1) + b_layer1)
    h_layer2 = tf.nn.relu(tf.matmul(h_layer1, W_layer2) + b_layer2)
    h_layer3 = tf.nn.relu(tf.matmul(h_layer2, W_layer3) + b_layer3)

    # output layer
    y = tf.matmul(h_layer3, o_layer) + o_bais
    prediction = tf.argmax(y[0])

    return x, y, prediction

def trainNetwork(usingTensorFlow):
    print()

    step = 0

    if usingTensorFlow:
        inputState, Qoutputs, prediction = createNetwork()

        targetQOutputs = tf.placeholder("float", [None, actions])
        loss = tf.reduce_mean(tf.square(tf.subtract(targetQOutputs, Qoutputs)))

        # train the model to minimise the loss
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        checkpoint = tf.train.get_checkpoint_state("model")
        if checkpoint and checkpoint.model_checkpoint_path:
            s = saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded the model:", checkpoint.model_checkpoint_path)
            step = int(os.path.basename(checkpoint.model_checkpoint_path).split('-')[1])
        else:
            print("Could not find old network weights")

    else:
        inputState = Qoutputs = prediction = None
        sess = TorchNetwork() # should be called network
        optimizer = optim.Adam(sess.parameters(), lr = 1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        criterion = nn.MSELoss()
        try:
            checkpoint = torch.load("./model/model.pt", map_location=torch.device('cpu'))
            sess.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            step = checkpoint['step']
            print("Successfully loaded the model:", "./model/model.pt")
        except:
            print("Could not find old saved model")

    print(time.ctime())
    run_time = 0

    episodes = 500
    epochs = 100
    # defines the rate at which epsilon should decrease
    e_downrate = 0.0075
    # greedy policy for selecting an action
    # the higher the value of e the higher the probability of an action being random.
    epsilon = 1.0

    while True:
        start_time = time.time()
        global won_games
        global lost_games
        global draw_games

        total_loss = 0
        GamesList = []
        if not usingTensorFlow:
            sess.eval()

        for i in range(episodes): # play 500 games
            completeGame = playaGame(usingTensorFlow, epsilon, sess, inputState, prediction, Qoutputs)
            GamesList.append(completeGame)

        for k in range(epochs):
            random.shuffle(GamesList)
            for i in GamesList:
                len_complete_game = len(i)
                for loop_in in range(len_complete_game):
                    j = i.pop()
                    currentState = j[0]
                    action = j[1][0]
                    reward = j[2][0]
                    nextState = j[3]

                    # Game end reward
                    if loop_in == 0:
                        game_reward = reward
                    else:
                        # obtain q values for next state using the network
                        if usingTensorFlow:
                            nextQ = sess.run(Qoutputs, feed_dict={inputState: [nextState]})
                        else:
                            nextQ = sess(torch.as_tensor(nextState, dtype=torch.float32)).detach().numpy()
                        maxNextQ = np.max(nextQ)
                        game_reward = GAMMA * (maxNextQ)

                    if usingTensorFlow:
                        targetQ = sess.run(Qoutputs, feed_dict={inputState: [currentState]})
                    else:
                        targetQ = sess(torch.as_tensor(currentState, dtype=torch.float32)).detach().numpy()

                    # -1 reward for all the illegal moves in the q value
                    targetQ[currentState!=0] = -1

                    if usingTensorFlow:
                        targetQ[0, action] = game_reward
                    else:
                        targetQ[action] = game_reward

                    # Train our network using the targetQ
                    if usingTensorFlow:
                        t_loss = sess.run([train_step, Qoutputs, loss], feed_dict={inputState: [currentState], targetQOutputs: targetQ})
                        total_loss += t_loss[2]
                    else:
                        sess.train()
                        optimizer.zero_grad()
                        Qoutputs = sess(torch.as_tensor(currentState, dtype=torch.float32).requires_grad_())
                        t_loss = criterion(torch.as_tensor(targetQ, dtype=torch.float32).requires_grad_(), Qoutputs)
                        t_loss.backward()
                        optimizer.step()
                        scheduler.step()
                        total_loss += t_loss

        step += 1
        time_diff = time.time() - start_time
        run_time += time_diff
        print("iteration {} completed with {} wins, {} losses {} draws, out of {} games played, e is {} \ncost is {} ,"
              " current_time is {}, time taken is {} , total time = {} hours \n".format(step, won_games, lost_games,
                draw_games, episodes, epsilon * 100, total_loss, time.ctime(), time_diff, (run_time) / 3600))
        start_time = time.time()
        total_loss = 0
        won_games = 0
        lost_games = 0
        draw_games = 0

        # decrease e value slowly.
        if epsilon > -0.2:
            epsilon -= e_downrate
        else:
            epsilon = random.choice([0.1, 0.05, 0.06, 0.07, 0.15, 0.03, 0.20, 0.25, 0.5, 0.4])

        if usingTensorFlow:
            saver.save(sess, "./model/model.ckpt", global_step=step)
        else:
            torch.save({'model':sess.state_dict(), 'optimizer':optimizer.state_dict(), 'scheduler':scheduler, 'step':step},"./model/model.pt")

# plays a game and returns a list with all states, actions and final reward.
def playaGame(usingTensorFlow, e, sess, inputState, prediction, Qoutputs):
    global won_games
    global lost_games
    global draw_games

    win_reward = 10
    loss_reward = -1
    draw_reward = 3
    invalid_reward = -10

    completeGameMemory = []
    myList = np.zeros(9)

    # randomly chose a turn 1 is ours -1 is opponent
    turn = random.choice([1, -1])

    # if opponent turn let him play and set the initial state
    if (turn == -1):
        initial_index = random.choice(range(9))
        if usingTensorFlow:
            best_index, _ = sess.run([prediction, Qoutputs],
                                 feed_dict={inputState: [np.copy(myList)]})
        else:
            best_index = np.argmax(sess(torch.as_tensor(myList, dtype=torch.float32)).detach().numpy())
        initial_index = random.choice([best_index, initial_index, best_index])
        myList[initial_index] = -1
        turn = turn * -1

    # until game over
    while (True):

        # create a memory which will hold the current initial state, the action thats taken, the reward the was recieved, the next state
        memory = []

        zero_indexes = np.where(myList==0)[0]
        # if no index is found which is free to place a move exit as the game completed with slight reward. better to draw then to lose right ?
        if len(zero_indexes) == 0:
            completeGameMemory[-1][2][0] = draw_reward
            draw_games += 1
            break

        # calculate the prediction from the network which can be later used as an action with some probability
        if usingTensorFlow:
            pred, _ = sess.run([prediction, Qoutputs], feed_dict={inputState: [myList]})
        else:
            pred = np.argmax(sess(torch.as_tensor(myList, dtype=torch.float32)).detach().numpy())

        # lets add the initial state to the current memory
        memory.append(np.copy(myList))

        # Lets pick an action with some probability, exploration and exploitation
        if random.random() > e:  # and isFalsePrediction == False: #expliotation
            action = pred
        else:  # exploration, explore with valid moves to save time.
            action = random.choice(zero_indexes)

        # lets add the action to the memory
        memory.append([action])

        # plays a wrong move
        if action not in zero_indexes:
            memory.append([invalid_reward])
            memory.append(np.copy(myList))
            completeGameMemory.append(memory)
            lost_games += 1
            break

        # update the board with the action taken
        myList[action] = 1

        # if after playing our move the game is completed then yay we deserve a reward and its the final state
        if (isGameOver(myList)):
            memory.append([win_reward])
            memory.append(np.copy(myList))
            completeGameMemory.append(memory)
            won_games += 1
            break

        # Now lets make a move for the opponent

        zero_indexes = np.where(myList==0)[0]
        # if opponent has no moves left that means that the last move was the final move and its a draw so some reward
        if len(zero_indexes) == 0:
            memory.append([draw_reward])
            memory.append(np.copy(myList))
            completeGameMemory.append(memory)
            draw_games += 1
            break

        # same as before, but since we are finding a move for the opponent we use the inverse board
        # to calculate the prediction
        myList_inverse = InverseBoard(myList)

        # almost same as before
        selectedRandomIndex = random.choice(zero_indexes)
        if usingTensorFlow:
            pred, _ = sess.run([prediction, Qoutputs], feed_dict={inputState: [myList_inverse]})
        else:
            pred = np.argmax(sess(torch.as_tensor(myList_inverse, dtype=torch.float32)).detach().numpy())
        isFalsePrediction = False if myList[pred] == 0 else True

        # we want opponent to play good sometimes and play bad sometimes so 33.33% ish probability
        if (isFalsePrediction == True):
            action = selectedRandomIndex
        else:
            action = random.choice([selectedRandomIndex, pred, pred, pred, pred])

        # update the board with opponents move
        myList[action] = -1

        # if after opponents move the game is done meaning opponent won, boo..
        if isGameOver(myList):
            memory.append([loss_reward])
            # final state
            memory.append(np.copy(myList))
            completeGameMemory.append(memory)
            lost_games += 1
            break

        # if no one won and game isn't done yet then lets continue the game
        memory.append([0])
        memory.append(np.copy(myList))
        completeGameMemory.append(memory)

    return completeGameMemory

if __name__ == "__main__":
    usingTensorFlow = False
    trainNetwork(usingTensorFlow)
