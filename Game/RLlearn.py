#Params to train on:
#Move list - backprop on this
#Predict best future piece moves
import torch
import torch.nn as nn
import torch.optim as optim
from pieces import *
from board import *
from game import *

global board
global piece_list

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 24)
        self.fc3 = nn.Linear(24, 8)
        self.fc4 = nn.Linear(8, 2)
        self.fc5 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = torch.softmax(self.fc5(x), dim=0)
        flattened = x.flatten()

        return flattened.view(2, -1)

def main():
    global board
    board = setup()
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
    # print(board.grid[6][1].move((6, 1), (7, 1), board))
    # print(board.grid[5][2].move((5, 2), (7, 1), board))
    board.display()

if __name__ == '__main__':
    global board
    main()
    module = NeuralNetwork()
    print(board.grid[6][1].history)
    tensor_history = torch.tensor(board.grid[6][1].history, dtype=torch.float32)
    epoch = 10
    optimizer = torch.optim.Adam(module.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()
    output = tensor_history
    target = torch.ones(2, 1) * 8

    for i in range(epoch):
        optimizer.zero_grad()
        output = module.forward(tensor_history)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(output)

    print(output.mean(dim = 1))