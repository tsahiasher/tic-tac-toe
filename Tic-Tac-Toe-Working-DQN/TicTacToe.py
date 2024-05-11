import numpy as np
needToWinGLOBAL = 3

class TicTacToeStatic:
    @staticmethod
    def available_moves(s):
        m = []
        length = len(s)
        for i in range(length):
            for j in range(length):
                if(s[i,j] == 0):
                    m.append((i,j))
        return m
    
    @staticmethod
    def Status(s, lastmove = None, cnt = None):
        '''if(len(s) != 3):
            TTT = TicTacToe(needToWinGLOBAL, len(s))
            return TTT.Status(s,lastmove,cnt)'''

        all = []
        for x in s.tolist():
            all.append(x)
        for x in [list(i) for i in zip(*s)]:
            all.append(x)

        all.append([s[0, 0], s[1, 1], s[2, 2]])
        all.append([s[2, 0], s[1, 1], s[0, 2]])

        e = 0

        if [1, 1, 1] in all:
            e = 1
        elif [-1, -1, -1] in all:
            e = -1
        else:
            for i in range(3):
                for j in range(3):
                    if(s[i, j] == 0):
                        e = None
        return e
    
    @staticmethod
    def nearest(s,r,c):
        mi = 999999999
        length = len(s)
        for i in range(length):
            for j in range(length):
                if(s[i][j]!=0):
                    mi = min(mi,abs(i-r)+abs(j-c))
        if(mi == 999999999):
            mi = 0
        return mi

    @staticmethod
    def getNTW():
        return needToWinGLOBAL

    @staticmethod
    def removecopies(s,m):
        boards = []
        copies = []
        newboards = []
        
        for i in range(len(m)):
            move = m[i]
            new_s = s.copy()
            r = move[0]
            c = move[1]
            new_s[r][c] = 10
            if(TicTacToeStatic.nearest(s,r,c)>2):
                copies.append(i)
            boards.append(new_s)
            new_s=None

        if(len(s)!=10):
            for i in range(len(m)):
                for j in range(len(m)):
                    if(i>=j or i in copies or j in copies):
                        continue
                    if(np.array_equal(boards[i],np.flipud(boards[j]))):
                        copies.append(j)
                    elif(np.array_equal(boards[i],np.fliplr(boards[j]))):
                        copies.append(j)
                    elif(np.array_equal(boards[i],np.rot90(boards[j]))):
                        copies.append(j)
                    elif(np.array_equal(boards[i],np.rot90(np.rot90(boards[j])))):
                        copies.append(j)
                    elif(np.array_equal(boards[i],np.rot90(np.rot90(np.rot90(boards[j]))))):
                        copies.append(j)
        
        for i in range(len(m)):
            if(i in copies):
                continue
            newboards.append(m[i])
        
        return newboards