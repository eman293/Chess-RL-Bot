from pieces import *

class Board:
    def __init__(self):
        self.size = 8
        self.grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]

    def display(self):
        for row in self.grid:
            print('|'.join(row))

    #returns a tuple of {bool, string, string} = {isPresent, color, name}
    def piece_present(self, x, y):
        return {self.grid[x][y] != ' ', self.grid[x][y].color, self.grid[x][y].name} if InBounds(x, y) else {False, None, None}

    def capture(self, x1, y1, x2, y2):
        if InBounds(x1, y1) and InBounds(x2, y2):
            if self.grid[x2][y2] != ' ':
                captured_piece = self.grid[x2][y2]
                self.grid[x2][y2] = self.grid[x1][y1]
                self.grid[x1][y1] = ' '
                return captured_piece
            else:
                raise ValueError("No piece to capture at the target position.")
        else:
            raise IndexError("Position out of bounds.")


