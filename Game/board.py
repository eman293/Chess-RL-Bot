from pieces import *

class Board:
    def __init__(self):
        self.size = 8
        self.grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]

    def display(self):
        for row in self.grid:
            print('|'.join(row))


