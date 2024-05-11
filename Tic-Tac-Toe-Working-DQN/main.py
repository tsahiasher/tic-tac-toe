from deep_reinforcement_learning import *
usingTensorFlow = False
from MonteCarloTreeSearchPlayer import MonteCarloTreeSearchPlayer

def print_board(board, winningSeq):
    row = ' '
    for i in range(9):
        if board[i] == 1:
            cell = 'x'
        elif board[i] == -1:
            cell = 'o'
        else:
            cell = ' '
        if i in winningSeq:
            cell = cell.upper()
        row += cell + ' '
        if (i+1) % 3 != 0:
            row += '| '
        else:
            if i != 8:
                row += ' \n-----------'
            print(row)
            row = ' '

def play():
    if usingTensorFlow:
        sess = tf.InteractiveSession()
        x , prediction, _ = createNetwork()
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("model")
        if checkpoint and checkpoint.model_checkpoint_path:
            s = saver.restore(sess,checkpoint.model_checkpoint_path)
            print("Successfully loaded the model:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
        graph = tf.get_default_graph()
    else:
        network = TorchNetwork()
        try:
            checkpoint = torch.load("./model/model.pt", map_location=torch.device('cpu'))
            network.load_state_dict(checkpoint['model'])
            network.eval()
            print("Successfully loaded the model:", "./model/model.pt")
        except:
            print("Could not find old saved model")
            exit()

    beginner = -1
    for g in range(4):
        beginner *= -1
        print('STARTING NEW GAME (#{})\n-------------'.format(g))
        print("Computer is X (1)")
        board = [0]*9
        winningSeq = []
        gameOver = False
        winner = 0
        currPlayer = beginner
        while not gameOver:
            if currPlayer==-1:
                print_board(board, winningSeq)
                print("{}'s turn:".format('X' if currPlayer==1 else 'O'))
                while True:
                    try:
                        move = int(input("Select cell to fill:\n012\n345\n678\ncell number: "))
                        if move < 0 or move > 8:
                            raise ValueError
                    except ValueError:
                        print("Oops! That was no valid number. Try again...")
                    else:
                        break
            else:
                if usingTensorFlow:
                    with graph.as_default():
                        data = (sess.run(tf.argmax(prediction.eval(session = sess,feed_dict={x:[board]}),1)))
                    move = data[0].item()
                else:
                    move = np.argmax(network(torch.as_tensor(board, dtype=torch.float32)).detach().numpy())

            if board[move]:
                continue
            else:
                board[move] = currPlayer

            gameOver, winner, winningSeq = game_status(board)
            currPlayer*=-1

        print('-------------\nGAME OVER!')
        print_board(board, winningSeq)
        if winner:
            print('winner {}'.format('X' if winner==1 else 'O'))
        else:
            print('Draw')
        print('-------------')

def playAgainstMCTS():

    MCTSPlayer = MonteCarloTreeSearchPlayer(-1)
    beginner = -1
    for g in range(100):
        beginner *= -1
        board = [0]*9
        gameOver = False
        winner = 0
        currPlayer = beginner
        winningSeq = []
        while not gameOver:
            if currPlayer==1:
                print_board(board, winningSeq)
                print("{}'s turn:".format('X' if currPlayer==1 else 'O'))
                while True:
                    try:
                        move = int(input("Select cell to fill:\n012\n345\n678\ncell number: "))
                        if move < 0 or move > 8:
                            raise ValueError
                    except ValueError:
                        print("Oops! That was no valid number. Try again...")
                    else:
                        break
            else:
                move = MCTSPlayer.make_a_move(board)

            if board[move]:
                continue
            else:
                board[move] = currPlayer

            gameOver, winner, winningSeq = game_status(board)
            currPlayer*=-1

        print('-------------\nGAME OVER!')
        print_board(board, winningSeq)
        if winner:
            print('winner {}'.format('X' if winner==1 else 'O'))
        else:
            print('Draw')
        print('-------------')

def DQNAgainstMCTS():

    DQNPlayer = TorchNetwork()
    try:
        checkpoint = torch.load("./model/model.pt", map_location=torch.device('cpu'))
        DQNPlayer.load_state_dict(checkpoint['model'])
        DQNPlayer.eval()
        print("Successfully loaded the model:", "./model/model.pt")
    except:
        print("Could not find old saved model")
        exit()

    MCTSPlayer = MonteCarloTreeSearchPlayer(-1)
    beginner = -1
    draw = 0
    MCTSWins = 0
    DQNWins = 0
    for g in range(100):
        beginner *= -1
        board = [0]*9
        gameOver = False
        winner = 0
        currPlayer = beginner
        while not gameOver:
            if currPlayer==-1:
                move = MCTSPlayer.make_a_move(board)
            else:
                move = np.argmax(DQNPlayer(torch.as_tensor(board, dtype=torch.float32)).detach().numpy())

            if board[move]:
                continue
            else:
                board[move] = currPlayer

            gameOver, winner, winningSeq = game_status(board)
            currPlayer*=-1

        if winner==1:
            DQNWins+=1
            print_board(board, winningSeq)
        elif winner==-1:
            MCTSWins+=1
        else:
            draw+=1

    print(f'DQN:{DQNWins}\nMCTS:{MCTSWins}\nDraw:{draw}')

if __name__ == '__main__':
    trainNetwork(usingTensorFlow)
    #play()
    #DQNAgainstMCTS()
    #playAgainstMCTS()

