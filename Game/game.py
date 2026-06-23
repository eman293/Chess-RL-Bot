from pieces import *
from board import *

global board
global piece_list

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
    piece_list = []
    for row in board.grid:
        for piece in row:
            if piece != ' ':
                piece_list.append(piece)

    print(board.grid[1][0].move((1, 0), (3, 0), board))
    print(board.grid[6][1].move((6, 1), (4, 1), board))
    print(board.grid[3][0].move((3, 0), (4, 1), board))
    print(board.grid[7][1].move((7, 1), (5, 2), board))
    print(board.grid[4][1].move((4, 1), (5, 1), board))
    print(board.grid[5][1].move((5, 1), (6, 1), board))
    print(board.grid[7][2].move((7, 2), (6, 1), board))
    board.display()

if __name__ == '__main__':
    main()