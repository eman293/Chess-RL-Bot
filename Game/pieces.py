from board import *

def InBounds(x1, y1):
    return (0 <= x1 < 8) and (0 <= y1 < 8)

#returns a tuple of {bool, string, string} = {isPresent, color, name}
def piece_present(board, x, y):
    return {board.grid[x][y] != ' ', board.grid[x][y].color, board.grid[x][y].name} if InBounds(x, y) else {False, None, None}

class Piece:
    def __init__(self, name, color):
        self.name = name
        self.color = color

    def __str__(self):
        return f"{self.color} {self.name}"

    #start_pos and end_pos are tuples of (row, col)
    def move(self, start_pos, end_pos, board):
        pass

class Pawn(Piece):
    def __init__(self, color):
        super().__init__("Pawn", color)

    def move(self, start_pos, end_pos, board):
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        if self.color == "W":
            if(start_col == end_col):
                if(end_row == 7):
                    #TODO HANDLE PROMOTION
                    raise NotImplementedError("Pawn promotion not implemented yet.")
                elif((start_row == 1 and end_row == 3 and not piece_present(board, 2, start_col)[0] and not piece_present(board, 3, start_col)[0])
                    or start_row + 1 == end_row and not piece_present(board, end_row, end_col)[0] and InBounds(end_row, end_col)):
                        return True
                else:
                    return False
            else:
                if(InBounds(end_row, end_col) and start_row + 1 == end_row and abs(start_col - end_col) == 1 and piece_present(board, end_row, end_col)[0]):
                    raise NotImplementedError("Pawn capture not implemented yet.")
                    return True
                elif(InBounds(end_row, end_col) and start_row + 1 == end_row and abs(start_col - end_col) == 1 and piece_present(board, end_row, end_col)[1] = "B"):
                    raise NotImplementedError("Pawn en passante not implemented yet.")
                    return True
                else:
                    return False

class Knight(Piece):
    def __init__(self, color):
        super().__init__("Knight", color)

    def move(self, start_pos, end_pos, board):
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        if((inBounds(end_row, end_col) and ((abs(start_row - end_row) == 2 and abs(start_col - end_col) == 1) 
            or (abs(start_row - end_row) == 1 and abs(start_col - end_col) == 2)))):
                if(piece_present(board, end_row, end_col)[1] != self.color):
                    raise NotImplementedError("Knight capture not implemented yet.")
                    return True
                elif(piece_present(board, end_row, end_col)[0] == False):
                    return True
                else:
                    return False
        else:
            return False

class Bishop(Piece):
    def __init__(self, color):
        super().__init__("Knight", color)

    def move(self, start_pos, end_pos, board):
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        if(inBounds(end_row, end_col) and abs(start_row - end_row) == abs(start_col - end_col)):
            for(i in range(1, abs(start_row - end_row))):
                if(piece_present(board, start_row + (i * (1 if end_row > start_row else -1)), start_col + (i * (1 if end_col > start_col else -1)))[0]):
                    return False
            if(piece_present(board, end_row, end_col)[1] != self.color):
                raise NotImplementedError("Bishop capture not implemented yet.")
                return True
            elif(piece_present(board, end_row, end_col)[0] == False):
                return True
            else:
                return False

class Rook(Piece):
    def __init__(self, color):
        super().__init__("Rook", color)

    def move(self, start_pos, end_pos, board):
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        if(inBounds(end_row, end_col) and (start_row == end_row or start_col == end_col)):
            if(start_row == end_row):
                for(i in range(1, abs(start_col - end_col))):
                    if(piece_present(board, start_row, start_col + (i * (1 if end_col > start_col else -1)))[0]):
                        return False
            else:
                for(i in range(1, abs(start_row - end_row))):
                    if(piece_present(board, start_row + (i * (1 if end_row > start_row else -1)), start_col)[0]):
                        return False
            if(piece_present(board, end_row, end_col)[1] != self.color):
                raise NotImplementedError("Rook capture not implemented yet.")
                return True
            elif(piece_present(board, end_row, end_col)[0] == False):
                return True
            else:
                return False

class Queen(Piece):
    def __init__(self, color):
        super().__init__("Queen", color)

    def move(self, start_pos, end_pos, board):
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        if(inBounds(end_row, end_col) and (start_row == end_row or start_col == end_col or abs(start_row - end_row) == abs(start_col - end_col))):
            if(start_row == end_row):
                for(i in range(1, abs(start_col - end_col))):
                    if(piece_present(board, start_row, start_col + (i * (1 if end_col > start_col else -1)))[0]):
                        return False
            elif(start_col == end_col):
                for(i in range(1, abs(start_row - end_row))):
                    if(piece_present(board, start_row + (i * (1 if end_row > start_row else -1)), start_col)[0]):
                        return False
            else:
                for(i in range(1, abs(start_row - end_row))):
                    if(piece_present(board, start_row + (i * (1 if end_row > start_row else -1)), start_col + (i * (1 if end_col > start_col else -1)))[0]):
                        return False
            if(piece_present(board, end_row, end_col)[1] != self.color):
                raise NotImplementedError("Queen capture not implemented yet.")
                return True
            elif(piece_present(board, end_row, end_col)[0] == False):
                return True
            else:
                return False

class King(Piece):
    def __init__(self, color):
        super().__init__("King", color)

    def move(self, start_pos, end_pos, board):
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        if(inBounds(end_row, end_col) and (abs(start_row - end_row) <= 1 and abs(start_col - end_col) <= 1)):
            if(piece_present(board, end_row, end_col)[1] != self.color):
                raise NotImplementedError("King capture not implemented yet.")
                return True
            elif(piece_present(board, end_row, end_col)[0] == False):
                return True
            else:
                return False

def main():
    board = Board()
    pawn = Pawn("White")
    pawn.move((1, 0), (3, 0), Board())
    print("success")

if __name__ == '__main__':
    main()
        