"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    xCount = 0
    oCount = 0
    for row in board:
        for cell in row:
            if cell == X:
                xCount += 1
            elif cell == O:
                oCount += 1
    
    if xCount > oCount:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    for row in range(3):
        for col in range(3):
            if board[row][col] == EMPTY:
                actions.add((row, col))
    
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    # check if action is available
    if board[action[0]][action[1]] != EMPTY:
        raise IndexError
    
    # make a copy of the board and make the move
    newBoard = []
    for i in range(3):
        arr = []
        for j in range(3):
            arr.append(board[i][j])
        newBoard.append(arr)

    newBoard[action[0]][action[1]] = player(board)

    return newBoard


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # check rows for winner
    for row in range(3):
        if board[row][0] != EMPTY and board[row][0] == board[row][1] == board[row][2]:
            return board[row][0]

    # check columns for winner    
    for col in range(3):
        if board[0][col] != EMPTY and board[0][col] == board[1][col] == board[2][col]:
            return board[0][col]

    # check diagonals for winner    
    if board[0][0] != EMPTY and board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]
    
    # check other diagonal for winner
    if board[0][2] != EMPTY and board[0][2] == board[1][1] == board[2][0]:
        return board[0][2]
    
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # check if there is a winner
    if winner(board) != None:
        return True
    
    # check if there are any empty cells
    for row in board:
        for cell in row:
            if cell == EMPTY:
                return False
    
    # if there is no winner and no empty cells, the game is over
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # check if game is over
    if terminal(board):
        return None
    
    if player(board) == X:
        v, action = maxFunction(board)
        return action
    else:
        v, action = minFunction(board)
        return action


def minFunction(board):
    """Return the min value of the board"""
    # check if game is over
    if terminal(board):
        return utility(board), None
    
    v = math.inf
    finalAction = None
    for action in actions(board):
        newValue, newAction = maxFunction(result(board, action))
        if (v > newValue):
            finalAction = action
            v = newValue
    
    return v, finalAction


def maxFunction(board):
    """Return the max value of the board"""
    # check if game is over
    if terminal(board):
        return utility(board), None
    
    v = -math.inf
    finalAction = None
    for action in actions(board):
        newValue, newAction = minFunction(result(board, action))
        if (v < newValue):
            finalAction = action
            v = newValue
        # v = max(v, minFunction(result(board, action)))
    
    return v, finalAction
