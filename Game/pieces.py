def InBounds(x1, y1):
    return (0 <= x1 < 8) and (0 <= y1 < 8)

class Piece:
    def __init__(self, name, color, icon, fen):
        self.name = name
        self.color = color
        self.icon = icon
        self.history = []
        self.fen = fen

    def __str__(self):
        return f"{self.color} {self.name}"

    def move(self, start_pos, end_pos, board, last_move=None):
        pass

class Pawn(Piece):
    def __init__(self, color):
        if color == 'W':
            super().__init__("Pawn", color, chr(0x265F), 'P')
        else:
            super().__init__("Pawn", color, chr(0x2659), 'p')

    def move(self, start_pos, end_pos, board, last_move=None):
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        if self.color == "W":
            if start_col == end_col:
                if end_row == 7:
                    board.update(start_row, start_col, end_row, end_col)
                    self.history.append((start_pos, end_pos))
                    return 'promotion'
                elif ((start_row == 1 and end_row == 3
                       and not board.piece_present(2, start_col)[0]
                       and not board.piece_present(3, start_col)[0])
                      or (start_row + 1 == end_row
                          and not board.piece_present(end_row, end_col)[0]
                          and InBounds(end_row, end_col))):
                    board.update(start_row, start_col, end_row, end_col)
                    self.history.append((start_pos, end_pos))
                    return True
                else:
                    raise RuntimeError("Invalid Move")
            else:
                if (InBounds(end_row, end_col)
                        and start_row + 1 == end_row
                        and abs(start_col - end_col) == 1):
                    if board.piece_present(end_row, end_col)[0] and board.piece_present(end_row, end_col)[1] != self.color:
                        board.capture(start_row, start_col, end_row, end_col)
                        self.history.append((start_pos, end_pos))
                        return True
                    elif last_move is not None:
                        lm_piece, lm_start, lm_end = last_move
                        lm_start_row, lm_start_col = lm_start
                        lm_end_row, lm_end_col = lm_end
                        if (isinstance(lm_piece, Pawn)
                                and lm_piece.color == "B"
                                and lm_start_row == 6 and lm_end_row == 4
                                and lm_end_col == end_col
                                and lm_end_row == start_row):
                            board.grid[start_row][end_col] = ' '
                            board.update(start_row, start_col, end_row, end_col)
                            self.history.append((start_pos, end_pos))
                            return True
                    raise RuntimeError("Invalid Move")
                else:
                    raise RuntimeError("Invalid Move")

        if self.color == "B":
            if start_col == end_col:
                if end_row == 0:
                    board.update(start_row, start_col, end_row, end_col)
                    self.history.append((start_pos, end_pos))
                    return 'promotion'
                elif ((start_row == 6 and end_row == 4
                       and not board.piece_present(5, start_col)[0]
                       and not board.piece_present(4, start_col)[0])
                      or (start_row - 1 == end_row
                          and not board.piece_present(end_row, end_col)[0]
                          and InBounds(end_row, end_col))):
                    board.update(start_row, start_col, end_row, end_col)
                    self.history.append((start_pos, end_pos))
                    return True
                else:
                    raise RuntimeError("Invalid Move")
            else:
                if (InBounds(end_row, end_col)
                        and start_row - 1 == end_row
                        and abs(start_col - end_col) == 1):
                    if board.piece_present(end_row, end_col)[0] and board.piece_present(end_row, end_col)[1] != self.color:
                        board.capture(start_row, start_col, end_row, end_col)
                        self.history.append((start_pos, end_pos))
                        return True
                    elif last_move is not None:
                        lm_piece, lm_start, lm_end = last_move
                        lm_start_row, lm_start_col = lm_start
                        lm_end_row, lm_end_col = lm_end
                        if (isinstance(lm_piece, Pawn)
                                and lm_piece.color == "W"
                                and lm_start_row == 1 and lm_end_row == 3
                                and lm_end_col == end_col
                                and lm_end_row == start_row):
                            board.grid[start_row][end_col] = ' '
                            board.update(start_row, start_col, end_row, end_col)
                            self.history.append((start_pos, end_pos))
                            return True
                    raise RuntimeError("Invalid Move")
                else:
                    raise RuntimeError("Invalid Move")

class Knight(Piece):
    def __init__(self, color):
        if color == 'W':
            super().__init__("Knight", color, chr(0x265E), 'N')
        else:
            super().__init__("Knight", color, chr(0x2658), 'n')

    def move(self, start_pos, end_pos, board, last_move=None):
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        if (InBounds(end_row, end_col) and ((abs(start_row - end_row) == 2 and abs(start_col - end_col) == 1)
            or (abs(start_row - end_row) == 1 and abs(start_col - end_col) == 2))):
            if board.piece_present(end_row, end_col)[0] == False:
                board.update(start_row, start_col, end_row, end_col)
                self.history.append((start_pos, end_pos))
                return True
            elif board.piece_present(end_row, end_col)[1] != self.color:
                board.capture(start_row, start_col, end_row, end_col)
                self.history.append((start_pos, end_pos))
                return True
            else:
                raise RuntimeError("Invalid Move")
        else:
            raise RuntimeError("Invalid Move")

class Rook(Piece):
    def __init__(self, color):
        if color == 'W':
            super().__init__("Rook", color, chr(0x265C), 'R')
        else:
            super().__init__("Rook", color, chr(0x2656), 'r')

    def move(self, start_pos, end_pos, board, last_move=None):
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        if not (InBounds(end_row, end_col) and (start_row == end_row or start_col == end_col)):
            raise RuntimeError("Invalid Move")

        if start_row == end_row:
            for i in range(1, abs(start_col - end_col)):
                if board.piece_present(start_row, start_col + (i * (1 if end_col > start_col else -1)))[0]:
                    raise RuntimeError("Invalid Move")
        else:
            for i in range(1, abs(start_row - end_row)):
                if board.piece_present(start_row + (i * (1 if end_row > start_row else -1)), start_col)[0]:
                    raise RuntimeError("Invalid Move")

        if board.piece_present(end_row, end_col)[0] == False:
            board.update(start_row, start_col, end_row, end_col)
            self.history.append((start_pos, end_pos))
            return True
        elif board.piece_present(end_row, end_col)[1] != self.color:
            board.capture(start_row, start_col, end_row, end_col)
            self.history.append((start_pos, end_pos))
            return True
        else:
            raise RuntimeError("Invalid Move")

class Bishop(Piece):
    def __init__(self, color):
        if color == 'W':
            super().__init__("Bishop", color, chr(0x265D), 'B')
        else:
            super().__init__("Bishop", color, chr(0x2657), 'b')

    def move(self, start_pos, end_pos, board, last_move=None):
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        if not (InBounds(end_row, end_col) and abs(start_row - end_row) == abs(start_col - end_col) and abs(start_row - end_row) > 0):
            raise RuntimeError("Invalid Move")

        for i in range(1, abs(start_row - end_row)):
            if board.piece_present(
                start_row + (i * (1 if end_row > start_row else -1)),
                start_col + (i * (1 if end_col > start_col else -1))
            )[0]:
                raise RuntimeError("Invalid Move")

        if board.piece_present(end_row, end_col)[0] == False:
            board.update(start_row, start_col, end_row, end_col)
            self.history.append((start_pos, end_pos))
            return True
        elif board.piece_present(end_row, end_col)[1] != self.color:
            board.capture(start_row, start_col, end_row, end_col)
            self.history.append((start_pos, end_pos))
            return True
        else:
            raise RuntimeError("Invalid Move")

class Queen(Piece):
    def __init__(self, color):
        if color == 'W':
            super().__init__("Queen", color, chr(0x265B), 'Q')
        else:
            super().__init__("Queen", color, chr(0x2655), 'q')

    def move(self, start_pos, end_pos, board, last_move=None):
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        if not (InBounds(end_row, end_col) and (start_row == end_row or start_col == end_col or abs(start_row - end_row) == abs(start_col - end_col))):
            raise RuntimeError("Invalid Move")

        if start_row == end_row:
            for i in range(1, abs(start_col - end_col)):
                if board.piece_present(start_row, start_col + (i * (1 if end_col > start_col else -1)))[0]:
                    raise RuntimeError("Invalid Move")
        elif start_col == end_col:
            for i in range(1, abs(start_row - end_row)):
                if board.piece_present(start_row + (i * (1 if end_row > start_row else -1)), start_col)[0]:
                    raise RuntimeError("Invalid Move")
        else:
            for i in range(1, abs(start_row - end_row)):
                if board.piece_present(
                    start_row + (i * (1 if end_row > start_row else -1)),
                    start_col + (i * (1 if end_col > start_col else -1))
                )[0]:
                    raise RuntimeError("Invalid Move")

        if board.piece_present(end_row, end_col)[0] == False:
            board.update(start_row, start_col, end_row, end_col)
            self.history.append((start_pos, end_pos))
            return True
        elif board.piece_present(end_row, end_col)[1] != self.color:
            board.capture(start_row, start_col, end_row, end_col)
            self.history.append((start_pos, end_pos))
            return True
        else:
            raise RuntimeError("Invalid Move")

class King(Piece):
    def __init__(self, color):
        if color == 'W':
            super().__init__("King", color, chr(0x265A), 'K')
        else:
            super().__init__("King", color, chr(0x2654), 'k')

    def move(self, start_pos, end_pos, board, last_move=None):
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        # Castling
        if (start_col == 4 and end_col in (2, 6)
                and start_row == end_row
                and len(self.history) == 0):
            row = start_row
            if end_col == 6:
                rook = board.grid[row][7]
                if (rook != ' ' and rook.name == 'Rook'
                        and len(rook.history) == 0
                        and not board.piece_present(row, 5)[0]
                        and not board.piece_present(row, 6)[0]):
                    board.update(row, 4, row, 6)
                    board.update(row, 7, row, 5)
                    self.history.append((start_pos, end_pos))
                    rook.history.append(((row, 7), (row, 5)))
                    return True
            elif end_col == 2:
                rook = board.grid[row][0]
                if (rook != ' ' and rook.name == 'Rook'
                        and len(rook.history) == 0
                        and not board.piece_present(row, 1)[0]
                        and not board.piece_present(row, 2)[0]
                        and not board.piece_present(row, 3)[0]):
                    board.update(row, 4, row, 2)
                    board.update(row, 0, row, 3)
                    self.history.append((start_pos, end_pos))
                    rook.history.append(((row, 0), (row, 3)))
                    return True
            raise RuntimeError("Invalid Move")

        # Prevent moving adjacent to the opposing king
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                adj_row, adj_col = end_row + dr, end_col + dc
                if InBounds(adj_row, adj_col):
                    adj = board.grid[adj_row][adj_col]
                    if adj != ' ' and adj.name == "King" and adj.color != self.color:
                        raise RuntimeError("Invalid Move: adjacent to opposing king")

        # Normal move
        if (InBounds(end_row, end_col)
                and abs(start_row - end_row) <= 1
                and abs(start_col - end_col) <= 1):
            if board.piece_present(end_row, end_col)[0] == False:
                board.update(start_row, start_col, end_row, end_col)
                self.history.append((start_pos, end_pos))
                return True
            elif board.piece_present(end_row, end_col)[1] != self.color:
                board.capture(start_row, start_col, end_row, end_col)
                self.history.append((start_pos, end_pos))
                return True
            else:
                raise RuntimeError("Invalid Move")
        else:
            raise RuntimeError("Invalid Move")