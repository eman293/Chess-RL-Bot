import pieces as pieces

class Board:
    def __init__(self):
        self.size = 8
        self.grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]

    def display(self):
        for i in range(len(self.grid) - 1, -1, -1):
            print('|'.join(row.icon if row != ' ' else ' ' for row in self.grid[i]))

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


