# Chess RL Bot

This is my Chess RL Website - utilizes ideas from Bertsekas et al., "Superior Computer Chess with
Model Predictive Control, Reinforcement Learning, and Rollout" (arXiv
2409.06477)

# Interface

First page allows you to play chess with someone else on the same device or against a bot

Second page allows you to upload a .pth file to play against a chess bot either playing black or white

# How to run: 

Get your local path to stockfish and change the stockfish path in game.py and rllearn.py

Install python version >= 13.3 and run pip install -r requirements.txt

Run s.ps1 to launch the website

# Features in Progress

- Invoke penalty for repetition of moves
- Game rules begin to be ignored after training for a high number of iterations
- Pretrain on existing dataset of chess positions before rl happens

# Architecture Documentation

- TODO