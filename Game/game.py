from pieces import *
from board import *

global board

def setup():
    global board
    board = Board()
    # Place pieces on the board
    board.grid[7][0] = Rook("B")
    board.grid[7][1] = Knight("B")
    board.grid[7][2] = Bishop("B")
    board.grid[7][3] = Queen("B")
    board.grid[7][4] = King("B")
    board.grid[7][5] = Bishop("B")
    board.grid[7][6] = Knight("B")
    board.grid[7][7] = Rook("B")
    for i in range(8):
        board.grid[6][i] = Pawn("B")

    board.grid[0][0] = Rook("W")
    board.grid[0][1] = Knight("W")
    board.grid[0][2] = Bishop("W")
    board.grid[0][3] = Queen("W")
    board.grid[0][4] = King("W")
    board.grid[0][5] = Bishop("W")
    board.grid[0][6] = Knight("W")
    board.grid[0][7] = Rook("W")
    for i in range(8):
        board.grid[1][i] = Pawn("W")

    return board

def main():
    setup()
    print(board.grid[1][0].move((1, 0), (3, 0), board))
    board.display()

if __name__ == '__main__':
    main()