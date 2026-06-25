import pieces as pieces
import copy

class Board:
    def __init__(self):
        self.size = 8
        self.grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]

    def display(self):
        rows = []
        for i in range(len(self.grid) - 1, -1, -1):
            rows.append([row.icon if row != ' ' else ' ' for row in self.grid[i]])

        for i in range(len(self.grid)):
            print('|'.join(row.icon if row != ' ' else ' ' for row in self.grid[i]))

        for i in range(len(self.grid)):
            print('|'.join(row.color if row != ' ' else ' ' for row in self.grid[i]))
        
        return rows

    #returns a tuple of {bool, string, string} = {isPresent, color, name}
    def piece_present(self, x, y):
        try:
            return [self.grid[x][y].icon != ' ', self.grid[x][y].color, self.grid[x][y].name]
        except:
            return [False, None, None]

    def capture(self, x1, y1, x2, y2):
        if pieces.InBounds(x1, y1) and pieces.InBounds(x2, y2):
            if self.grid[x2][y2] != ' ':
                captured_piece = self.grid[x2][y2]
                self.grid[x2][y2] = self.grid[x1][y1]
                self.grid[x1][y1] = ' '
                return captured_piece
            else:
                raise ValueError("No piece to capture at the target position.")
        else:
            raise IndexError("Position out of bounds.")

    def update(self, x1, y1, x2, y2):
        if pieces.InBounds(x1, y1) and pieces.InBounds(x2, y2):
            self.grid[x2][y2] = self.grid[x1][y1]
            self.grid[x1][y1] = ' '
        else:
            raise IndexError("Position out of bounds.")
    
    def promote(self, x, y, new_piece):
        if pieces.InBounds(x, y):
            self.grid[x][y] = new_piece
        else:
            raise IndexError("Position out of bounds.")

def is_in_check(board, color, last_move=None):
    king_pos = None
    for r in range(8):
        for c in range(8):
            piece = board.grid[r][c]
            if piece != ' ' and piece.name == 'King' and piece.color == color:
                king_pos = (r, c)
                break

    if king_pos is None:
        return False

    king_row, king_col = king_pos
    opponent = 'B' if color == 'W' else 'W'

    for r in range(8):
        for c in range(8):
            piece = board.grid[r][c]
            if piece == ' ' or piece.color != opponent:
                continue

            if piece.name == 'Pawn':
                attack_row = r + (1 if piece.color == 'W' else -1)
                if attack_row == king_row and abs(c - king_col) == 1:
                    print(f"Pawn at ({r},{c}) color={piece.color} threatens {color} king at {king_pos}")
                    return True
            else:
                try:
                    test_board = copy.deepcopy(board)
                    test_piece = test_board.grid[r][c]
                    test_piece.move((r, c), king_pos, test_board, last_move)
                    print(f"{piece.name} at ({r},{c}) color={piece.color} threatens {color} king at {king_pos}")
                    return True
                except:
                    pass

    return False


def is_in_checkmate(board, color, last_move=None):
    for r in range(8):
        for c in range(8):
            piece = board.grid[r][c]
            if piece != ' ' and piece.color == color:
                for tr in range(8):
                    for tc in range(8):
                        try:
                            test_board = copy.deepcopy(board)
                            test_piece = test_board.grid[r][c]
                            test_piece.move((r, c), (tr, tc), test_board, last_move)
                            if not is_in_check(test_board, color):
                                return False
                        except:
                            pass
    return True

def is_stalemate(board, color, last_move=None):
    if is_in_check(board, color, last_move):
        return False  # if in check it's not stalemate
    
    # If no legal moves exist and not in check, it's stalemate
    for r in range(8):
        for c in range(8):
            piece = board.grid[r][c]
            if piece != ' ' and piece.color == color:
                for tr in range(8):
                    for tc in range(8):
                        try:
                            test_board = copy.deepcopy(board)
                            test_piece = test_board.grid[r][c]
                            test_piece.move((r, c), (tr, tc), test_board, last_move)
                            if not is_in_check(test_board, color, last_move):
                                return False  # found a legal move
                        except:
                            pass
    return True


