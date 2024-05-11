import logging
import random
from time import time
import numpy as np
import matplotlib.pyplot as plt
usingTensorFlow = False
if usingTensorFlow:
    import tensorflow as tf
import players
from game import Game
import copy

action_rot90 = [6,3,0,7,4,1,8,5,2]
action_fliplr = [2,1,0,5,4,3,8,7,6]

def rot90(mem):
    mem['state'] = np.rot90(mem['state'].reshape((3,3))).reshape(9)
    mem['next_state'] = np.rot90(mem['next_state'].reshape((3,3))).reshape(9)
    mem['action'] = action_rot90[mem['action']]
    return mem

def flip(mem):
    mem['state'] = np.fliplr(mem['state'].reshape((3,3))).reshape(9)
    mem['next_state'] = np.fliplr(mem['next_state'].reshape((3,3))).reshape(9)
    mem['action'] = action_fliplr[mem['action']]
    return mem

def augment(p, mem):
    temp_mem = copy.deepcopy(mem)
    copy_mem = copy.deepcopy(temp_mem)
    for j in range(2):
        for i in range(3):
            copy_mem = rot90(copy_mem)
            p.add_to_memory(copy.deepcopy(copy_mem))
        copy_mem = flip(temp_mem)

def train():
    costs = []  # this will store the costs, so we can plot them later
    r1 = []  # same, but for the players total rewards
    r2 = []
    random.seed(int(time()*1000))
    if usingTensorFlow:
        tf.reset_default_graph()
    logging.getLogger().setLevel(logging.WARN)

    # Initialize players
    p1 = players.QPlayer([100,160,160,100],
                         learning_batch_size=150, batches_to_q_target_switch=1000,
                         gamma=0.95, tau=0.95, memory_size=100000)
    p1.name = 'Q'
    p2 = players.Novice()
    p2.name = 'N'
    total_rewards = {p1.name: 0, p2.name: 0}
    lr = 0.001

    # Start playing
    num_of_games = 50000
    for g in range(1,num_of_games+1):
        game = Game(p1,p2) if g%2==0 else Game(p2,p1)  # make sure both players play X and O
        last_phases = {p1.name: None, p2.name: None}  # will be used to store the last state a player was in
        while not game.game_status()['game_over']:
            # If this is not the first move, store in memory the transition from the last state
            # the active player saw to this one
            state = np.copy(game.board)
            if last_phases[game.active_player().name] is not None:
                memory_element = last_phases[game.active_player().name]
                memory_element['next_state'] = state
                memory_element['game_over'] = False
                game.active_player().add_to_memory(memory_element)
                if game.active_player().name=='Q':
                    augment(game.active_player(), memory_element)
            # Calculate annealed epsilon
            if g <= num_of_games // 2:
                max_eps = 0.6
            elif g <= num_of_games // 1:
                max_eps = 0.01
            else:
                max_eps = 0.001
            min_eps = 0.01 if g <= num_of_games // 1 else 0.0
            eps = round(max(max_eps - round(g*(max_eps-min_eps)/num_of_games, 3), min_eps), 3)
            if not g%20000:
                lr = lr / 10.0

            # Play and receive reward
            action = int(game.active_player().select_cell(state, epsilon=eps))
            play_status = game.play(action)
            game_over = play_status['game_over']
            if play_status['invalid_move']:
                r = game.invalid_move_reward
            elif game_over:
                if play_status['winner'] == 0:
                    r = game.tie_reward
                else:
                    r = game.winning_reward
            else:
                r = 0

            # Store the current state in temporary memory
            last_phases[game.active_player().name] = {'state': state,
                                                      'action': action,
                                                      'reward': r}
            total_rewards[game.active_player().name] += r

            # Activate learning procedure
            cost = game.active_player().learn(learning_rate=lr)
            if cost is not None:
                costs.append(cost)

            # Next player's turn, if game hasn't ended
            if not game_over:
                game.next_player()

        # Adding last phase for winning (active) player
        memory_element = last_phases[game.active_player().name]
        memory_element['next_state'] = np.zeros(9)
        memory_element['game_over'] = True
        game.active_player().add_to_memory(memory_element)
        if game.active_player().name=='Q':
            augment(game.active_player(), memory_element)

        # Adding last phase for losing (inactive) player
        memory_element = last_phases[game.inactive_player().name]
        memory_element['next_state'] = np.zeros(9)
        memory_element['game_over'] = True
        memory_element['reward'] = game.losing_reward
        game.inactive_player().add_to_memory(memory_element)
        if game.inactive_player().name=='Q':
            augment(game.inactive_player(), memory_element)

        # Print statistics
        if g % 100 == 0:
            print('Game: {g} | Number of Trainings: {t} | Epsilon: {e} | Average Rewards - {p1}: {r1}, {p2}: {r2}'
                  .format(g=g, p1=p1.name, r1=total_rewards[p1.name]/100.0,
                          p2=p2.name, r2=total_rewards[p2.name]/100.0,
                          t=len(costs), e=eps))
            r1.append(total_rewards[p1.name]/100.0)
            r2.append(total_rewards[p2.name]/100.0)
            total_rewards = {p1.name: 0, p2.name: 0}

    # Save trained model and shutdown Tensorflow sessions
    if usingTensorFlow:
        p1.save('./models/q.ckpt')
        for pp in [p1,p2]:
            pp.shutdown()
    else:
        p1.save('./models/q.pt')

    # Plot graphs
    plt.scatter(range(len(costs)),costs)
    plt.show()
    plt.scatter(range(len(r1)),r1,c='g')
    plt.show()
    plt.scatter(range(len(r2)), r2, c='r')
    plt.show()


def play():
    random.seed(int(time()))
    p1 = players.QPlayer([100,160,160,100], learning_batch_size=100, gamma=0.95, tau=0.95,
                         batches_to_q_target_switch=100, memory_size=100000)
    if usingTensorFlow:
        p1.restore('./models/q.ckpt')
    else:
        p1.restore('./models/q.pt')
    p2 = players.Human()
    for g in range(4):
        print('STARTING NEW GAME (#{})\n-------------'.format(g))
        if g%2==0:
            game = Game(p1,p2)
            print("Computer is X (1)")
        else:
            game = Game(p2,p1)
            print("Computer is O (-1)")
        while not game.game_status()['game_over']:
            if isinstance(game.active_player(), players.Human):
                game.print_board()
                print("{}'s turn:".format(game.current_player))
            state = np.copy(game.board)
            # Force Q-Network to select different starting positions if it plays first
            if np.count_nonzero(game.board) > 0 or not isinstance(game.active_player(),players.QPlayer):
                action = int(game.active_player().select_cell(state,epsilon=0.0))
            else:
                action = random.randint(0,8)
            game.play(action)
            if not game.game_status()['game_over']:
                game.next_player()
        print('-------------\nGAME OVER!')
        game.print_board()
        print(game.game_status())
        print('-------------')
        
#train()
play()

