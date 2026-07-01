import pieces as pieces
import copy

def _in_bounds(r, c):
    return 0 <= r < 8 and 0 <= c < 8

class Board:
    def __init__(self):
        self.size = 8
        self.grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]

    def display(self):
        rows = []
        for i in range(len(self.grid)):
            row = []
            for piece in self.grid[i]:
                if piece != ' ':
                    row.append({'fen': piece.fen, 'color': piece.color})
                else:
                    row.append({'fen': ' ', 'color': None})
            rows.append(row)

        for i in range(len(self.grid) - 1, -1, -1):
            print('|'.join(row.icon if row != ' ' else ' ' for row in self.grid[i]))
            
        return rows

    def piece_present(self, x, y):
        try:
            p = self.grid[x][y]
            if p == ' ':
                return [False, None, None]
            return [True, p.color, p.name]
        except:
            return [False, None, None]

    def capture(self, x1, y1, x2, y2):
        if pieces.InBounds(x1, y1) and pieces.InBounds(x2, y2):
            if self.grid[x2][y2] != ' ':
                captured = self.grid[x2][y2]
                self.grid[x2][y2] = self.grid[x1][y1]
                self.grid[x1][y1] = ' '
                return captured
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
    """Fast geometry-only check detection — no deepcopy, no piece.move() calls."""
    king_pos = None
    for r in range(8):
        for c in range(8):
            p = board.grid[r][c]
            if p != ' ' and p.name == 'King' and p.color == color:
                king_pos = (r, c)
                break
        if king_pos:
            break

    if king_pos is None:
        return False

    kr, kc = king_pos
    opp = 'B' if color == 'W' else 'W'

    # Pawn attacks — from which direction would an opponent pawn attack the king?
    pawn_attack_row_offset = -1 if opp == 'W' else 1  # opp pawn attacks downward/upward
    for dc in (-1, 1):
        r, c = kr + pawn_attack_row_offset, kc + dc
        if _in_bounds(r, c):
            p = board.grid[r][c]
            if p != ' ' and p.name == 'Pawn' and p.color == opp:
                return True

    # Knight attacks
    for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
        r, c = kr + dr, kc + dc
        if _in_bounds(r, c):
            p = board.grid[r][c]
            if p != ' ' and p.name == 'Knight' and p.color == opp:
                return True

    # Rook / Queen on ranks & files
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        r, c = kr + dr, kc + dc
        while _in_bounds(r, c):
            p = board.grid[r][c]
            if p != ' ':
                if p.color == opp and p.name in ('Rook', 'Queen'):
                    return True
                break
            r += dr; c += dc

    # Bishop / Queen on diagonals
    for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        r, c = kr + dr, kc + dc
        while _in_bounds(r, c):
            p = board.grid[r][c]
            if p != ' ':
                if p.color == opp and p.name in ('Bishop', 'Queen'):
                    return True
                break
            r += dr; c += dc

    # King adjacency
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r, c = kr + dr, kc + dc
            if _in_bounds(r, c):
                p = board.grid[r][c]
                if p != ' ' and p.name == 'King' and p.color == opp:
                    return True

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
        return False
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
                                return False
                        except:
                            pass
    return True