"""
rllearn.py
==========

Off-policy Q-learning (DQN-style) reinforcement learning bot for the chess
engine defined in pieces.py / board.py.

Why not the `mpcrl` package?
-----------------------------
`mpcrl` (https://mpc-reinforcement-learning.readthedocs.io) implements RL on
top of Model Predictive Control for *continuous-state Linear Time-Invariant*
systems: s_{k+1} = A s_k + B a_k, with a quadratic cost and a CasADi-based
MPC solver providing the Q/V-function approximation. A chess board has no
such linear dynamics, no continuous action space, and a different legal
action set at every position, so the library's `LstdQLearningAgent` /
`Mpc` machinery has nothing to attach to here.

What we use instead from Bertsekas et al., "Superior Computer Chess with
Model Predictive Control, Reinforcement Learning, and Rollout" (arXiv
2409.06477):
  - The paper's "MPC-MC" framing treats the true game as a one-player
    sequential decision problem by folding the opponent's reply into the
    environment's transition function: x_{k+1} = f(x_k, u_k, w_k), where
    w_k is the move of a "nominal opponent" engine. We use Stockfish as
    that nominal opponent during self-play training games, so each
    Q-learning transition is (state, agent_move, reward, state_after_
    opponent_reply) -- a clean single-agent MDP, exactly matching the
    paper's deterministic/stochastic MPC-MC framing.
  - The paper's "position evaluator" engine is used to score resulting
    positions. We use Stockfish the same way, except instead of doing a
    1-2 ply brute-force lookahead with it (which would be extremely slow
    in pure Python against this exception/deepcopy-based move validator),
    we fold its centipawn evaluation into the *reward signal* that trains
    a learned Q-network. The Q-network then amortizes that lookahead
    information into a fast, single forward-pass policy.

Reward design (off-policy Q-learning target uses these per-ply rewards):
  - Large positive reward for delivering checkmate (win), smaller positive
    reward for a draw (stalemate) -- wins are valued much more than draws.
  - Large negative reward for being checkmated (loss).
  - Small bonus for putting the opponent in check / small penalty for
    being put in check ourselves (encourages king safety + initiative).
  - Reward proportional to net material swing (captures for us minus
    pieces lost) -- minimizes piece-value loss.
  - Reward proportional to the change in Stockfish's centipawn evaluation
    of the position from the agent's perspective -- maximizes positional
    evaluation.

Dependencies: `pip install torch` (required). `pip install stockfish`
(optional -- without it, training still runs using a random legal-move
opponent and skips the positional-evaluation reward term).
"""

import os
import copy
import math
import random
import threading
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from pieces import Pawn, Knight, Bishop, Rook, Queen, King, InBounds
from board import Board, is_in_check, is_in_checkmate, is_stalemate

try:
    from stockfish import Stockfish
except ImportError:
    Stockfish = None


# ─────────────────────────────────────────────────────────────────────────
# Board <-> tensor / action encoding
# ─────────────────────────────────────────────────────────────────────────

PIECE_TYPES = ['Pawn', 'Knight', 'Bishop', 'Rook', 'Queen', 'King']
PIECE_VALUES = {'Pawn': 1, 'Knight': 3, 'Bishop': 3, 'Rook': 5, 'Queen': 9, 'King': 0}
NUM_ACTIONS = 64 * 64  # (from_square, to_square); promotion is auto-queen


def other_color(color):
    return 'B' if color == 'W' else 'W'


def board_to_tensor(board, side_to_move):
    """12 piece planes (6 types x 2 colors) + 1 side-to-move plane, all 8x8."""
    planes = torch.zeros(13, 8, 8, dtype=torch.float32)
    for r in range(8):
        for c in range(8):
            piece = board.grid[r][c]
            if piece == ' ':
                continue
            type_idx = PIECE_TYPES.index(piece.name)
            color_offset = 0 if piece.color == 'W' else 6
            planes[type_idx + color_offset, r, c] = 1.0
    planes[12, :, :] = 1.0 if side_to_move == 'W' else 0.0
    return planes


def encode_action(from_pos, to_pos):
    f = from_pos[0] * 8 + from_pos[1]
    t = to_pos[0] * 8 + to_pos[1]
    return f * 64 + t


def decode_action(idx):
    f, t = divmod(idx, 64)
    return (f // 8, f % 8), (t // 8, t % 8)


def material_balance(board, color):
    """Net material (own total value - opponent total value) for `color`."""
    own, opp = 0, 0
    for r in range(8):
        for c in range(8):
            piece = board.grid[r][c]
            if piece == ' ':
                continue
            val = PIECE_VALUES.get(piece.name, 0)
            if piece.color == color:
                own += val
            else:
                opp += val
    return own - opp


# ─────────────────────────────────────────────────────────────────────────
# Legal move generation
#
# Re-uses the existing Piece.move() validators (so all your castling / en
# passant / check rules are respected automatically) but restricts the
# candidate target squares per piece type first, instead of brute-forcing
# all 64 squares -- this cuts the number of expensive deepcopy() calls
# substantially, which matters a lot for training throughput.
# ─────────────────────────────────────────────────────────────────────────

def _pawn_candidates(r, c, color):
    direction = 1 if color == 'W' else -1
    start_row = 1 if color == 'W' else 6
    candidates = [(r + direction, c), (r + direction, c - 1), (r + direction, c + 1)]
    if r == start_row:
        candidates.append((r + 2 * direction, c))
    return [(rr, cc) for rr, cc in candidates if InBounds(rr, cc)]


def _knight_candidates(r, c):
    offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    return [(r + dr, c + dc) for dr, dc in offsets if InBounds(r + dr, c + dc)]


def _sliding_candidates(r, c, directions):
    out = []
    for dr, dc in directions:
        for i in range(1, 8):
            rr, cc = r + dr * i, c + dc * i
            if not InBounds(rr, cc):
                break
            out.append((rr, cc))
    return out


def _bishop_candidates(r, c):
    return _sliding_candidates(r, c, [(-1, -1), (-1, 1), (1, -1), (1, 1)])


def _rook_candidates(r, c):
    return _sliding_candidates(r, c, [(-1, 0), (1, 0), (0, -1), (0, 1)])


def _queen_candidates(r, c):
    return _bishop_candidates(r, c) + _rook_candidates(r, c)


def _king_candidates(r, c):
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if InBounds(r + dr, c + dc):
                out.append((r + dr, c + dc))
    out.append((r, c + 2))  # kingside castle target
    out.append((r, c - 2))  # queenside castle target
    return [pos for pos in out if InBounds(*pos)]


_CANDIDATE_FUNCS = {
    'Pawn': lambda r, c, color: _pawn_candidates(r, c, color),
    'Knight': lambda r, c, color: _knight_candidates(r, c),
    'Bishop': lambda r, c, color: _bishop_candidates(r, c),
    'Rook': lambda r, c, color: _rook_candidates(r, c),
    'Queen': lambda r, c, color: _queen_candidates(r, c),
    'King': lambda r, c, color: _king_candidates(r, c),
}


def get_legal_moves(board, color, last_move=None):
    """All (from_pos, to_pos) pairs that are legal for `color`, i.e. they
    succeed via the piece's own move() validator AND don't leave color's
    own king in check."""
    legal = []
    for r in range(8):
        for c in range(8):
            piece = board.grid[r][c]
            if piece == ' ' or piece.color != color:
                continue
            for (tr, tc) in _CANDIDATE_FUNCS[piece.name](r, c, color):
                try:
                    test_board = copy.deepcopy(board)
                    test_piece = test_board.grid[r][c]
                    test_piece.move((r, c), (tr, tc), test_board, last_move)
                except Exception:
                    continue
                if is_in_check(test_board, color, last_move):
                    continue
                legal.append(((r, c), (tr, tc)))
    return legal


def fresh_board():
    """Standard starting position, independent of game.py so this module
    can also be run/tested standalone."""
    board = Board()
    board.grid[0][0] = Rook("W"); board.grid[0][1] = Knight("W"); board.grid[0][2] = Bishop("W")
    board.grid[0][3] = Queen("W"); board.grid[0][4] = King("W"); board.grid[0][5] = Bishop("W")
    board.grid[0][6] = Knight("W"); board.grid[0][7] = Rook("W")
    for i in range(8):
        board.grid[1][i] = Pawn("W")

    board.grid[7][0] = Rook("B"); board.grid[7][1] = Knight("B"); board.grid[7][2] = Bishop("B")
    board.grid[7][3] = Queen("B"); board.grid[7][4] = King("B"); board.grid[7][5] = Bishop("B")
    board.grid[7][6] = Knight("B"); board.grid[7][7] = Rook("B")
    for i in range(8):
        board.grid[6][i] = Pawn("B")
    return board


# ─────────────────────────────────────────────────────────────────────────
# Stockfish helpers (nominal opponent + position evaluator)
# ─────────────────────────────────────────────────────────────────────────

def local_board_to_fen(board, side_to_move, last_move=None):
    """Standalone FEN builder (board row r -> FEN rank r+1)."""
    rows = []
    for r in range(7, -1, -1):
        row_str, empty = "", 0
        for c in range(8):
            piece = board.grid[r][c]
            if piece == ' ':
                empty += 1
            else:
                if empty:
                    row_str += str(empty)
                    empty = 0
                row_str += piece.fen
        if empty:
            row_str += str(empty)
        rows.append(row_str)
    placement = "/".join(rows)

    stm = 'w' if side_to_move == 'W' else 'b'

    castling = ""
    wk = board.grid[0][4]
    if wk != ' ' and wk.name == 'King' and wk.color == 'W' and len(wk.history) == 0:
        wr_r, wr_l = board.grid[0][7], board.grid[0][0]
        if wr_r != ' ' and wr_r.name == 'Rook' and wr_r.color == 'W' and len(wr_r.history) == 0:
            castling += "K"
        if wr_l != ' ' and wr_l.name == 'Rook' and wr_l.color == 'W' and len(wr_l.history) == 0:
            castling += "Q"
    bk = board.grid[7][4]
    if bk != ' ' and bk.name == 'King' and bk.color == 'B' and len(bk.history) == 0:
        br_r, br_l = board.grid[7][7], board.grid[7][0]
        if br_r != ' ' and br_r.name == 'Rook' and br_r.color == 'B' and len(br_r.history) == 0:
            castling += "k"
        if br_l != ' ' and br_l.name == 'Rook' and br_l.color == 'B' and len(br_l.history) == 0:
            castling += "q"
    if not castling:
        castling = "-"

    en_passant = "-"
    if last_move is not None:
        lm_piece, (fr, fc), (tr, _tc) = last_move
        if lm_piece.name == 'Pawn' and abs(fr - tr) == 2:
            mid_row = (fr + tr) // 2
            en_passant = f"{chr(ord('a') + fc)}{mid_row + 1}"

    return f"{placement} {stm} {castling} {en_passant} 0 1"


def uci_to_positions(uci):
    from_col, from_row = ord(uci[0]) - ord('a'), int(uci[1]) - 1
    to_col, to_row = ord(uci[2]) - ord('a'), int(uci[3]) - 1
    return (from_row, from_col), (to_row, to_col)


def get_stockfish_move(stockfish, board, color, last_move=None):
    if stockfish is None:
        return None
    try:
        stockfish.set_fen_position(local_board_to_fen(board, color, last_move), do_validation=False)
        uci = stockfish.get_best_move()
    except Exception:
        return None
    return uci_to_positions(uci) if uci else None


def get_stockfish_eval(stockfish, board, color, last_move=None):
    """Centipawn evaluation from `color`'s perspective (None if unavailable)."""
    if stockfish is None:
        return None
    try:
        stockfish.set_fen_position(local_board_to_fen(board, color, last_move), do_validation=False)
        result = stockfish.get_evaluation()
    except Exception:
        return None
    if result['type'] == 'cp':
        return float(result['value'])
    return 10000.0 if result['value'] > 0 else -10000.0


# ─────────────────────────────────────────────────────────────────────────
# Q-network + replay buffer
# ─────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, num_actions=NUM_ACTIONS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512), nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_legal_actions):
        self.buffer.append((state, action, reward, next_state, done, next_legal_actions))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────
# Off-policy Q-learning agent (DQN: epsilon-greedy behavior, greedy target)
# ─────────────────────────────────────────────────────────────────────────

class ChessQLearningAgent:
    def __init__(self, color='W', lr=1e-3, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995, batch_size=64,
                 buffer_size=20000, win_reward=20.0, draw_reward=2.0,
                 loss_reward=-20.0, check_bonus=0.5, check_penalty=0.5,
                 material_weight=1.0, eval_weight=0.3, device=None):
        self.color = color
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Reward shaping weights (see module docstring)
        self.win_reward = win_reward
        self.draw_reward = draw_reward
        self.loss_reward = loss_reward
        self.check_bonus = check_bonus
        self.check_penalty = check_penalty
        self.material_weight = material_weight
        self.eval_weight = eval_weight

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)

    def select_action(self, board, legal_moves, training=True):
        if not legal_moves:
            return None
        if training and random.random() < self.epsilon:
            from_pos, to_pos = random.choice(legal_moves)
        else:
            state_tensor = board_to_tensor(board, self.color).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).squeeze(0)
            best_move, best_q = None, -float('inf')
            for (fp, tp) in legal_moves:
                q = q_values[encode_action(fp, tp)].item()
                if q > best_q:
                    best_q, best_move = q, (fp, tp)
            from_pos, to_pos = best_move
        return from_pos, to_pos, encode_action(from_pos, to_pos)

    def compute_reward(self, opp_in_check, agent_in_check, material_before,
                        material_after, eval_before, eval_after):
        reward = 0.0
        if opp_in_check:
            reward += self.check_bonus
        if agent_in_check:
            reward -= self.check_penalty
        reward += self.material_weight * (material_after - material_before)
        if eval_before is not None and eval_after is not None:
            delta = max(-2.0, min(2.0, (eval_after - eval_before) / 100.0))
            reward += self.eval_weight * delta
        return reward

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones, next_legal_lists = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_all = self.target_net(next_states)
            max_next_q = torch.zeros(len(batch), device=self.device)
            for i, legal in enumerate(next_legal_lists):
                if legal and dones[i].item() == 0:
                    legal_idx = torch.tensor(legal, dtype=torch.long, device=self.device)
                    max_next_q[i] = next_q_all[i, legal_idx].max()
            targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'epsilon': self.epsilon,
            'color': self.color,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.color = checkpoint.get('color', self.color)


# ─────────────────────────────────────────────────────────────────────────
# Self-play episode: agent vs. Stockfish-as-nominal-opponent
#
# Each agent-turn transition is (state_t, action, reward, state_{t+1}) where
# state_{t+1} is the position *after the opponent's reply*, matching the
# paper's f(x_k, u_k, w_k) transition -- so this is a clean single-agent
# MDP suitable for off-policy Q-learning.
# ─────────────────────────────────────────────────────────────────────────

def run_episode(agent, stockfish, opponent_color, max_plies=120, use_eval=True):
    board = fresh_board()
    last_move = None
    state_tensor = board_to_tensor(board, agent.color)
    total_reward = 0.0
    outcome = 'draw'

    for _ in range(max_plies):
        # board.display()
        # print("====================================")
        legal = get_legal_moves(board, agent.color, last_move)
        if not legal:
            if is_in_check(board, agent.color, last_move):
                outcome = 'loss-illegal_move'
            break

        material_before = material_balance(board, agent.color)
        eval_before = get_stockfish_eval(stockfish, board, agent.color, last_move) if use_eval else None

        from_pos, to_pos, action_idx = agent.select_action(board, legal, training=True)
        piece = board.grid[from_pos[0]][from_pos[1]]
        result = piece.move(from_pos, to_pos, board, last_move)
        if result == 'promotion':
            board.promote(to_pos[0], to_pos[1], Queen(agent.color))
        last_move = (board.grid[to_pos[0]][to_pos[1]], from_pos, to_pos)

        opp_in_check = is_in_check(board, opponent_color, last_move)
        opp_mated = opp_in_check and is_in_checkmate(board, opponent_color, last_move)

        if opp_mated:
            reward = agent.win_reward
            agent.memory.push(state_tensor, action_idx, reward,
                               board_to_tensor(board, agent.color), True, [])
            total_reward += reward
            outcome = 'win'
            break

        opp_stalemate = (not opp_in_check) and is_stalemate(board, opponent_color, last_move)
        if opp_stalemate:
            reward = agent.draw_reward
            agent.memory.push(state_tensor, action_idx, reward,
                               board_to_tensor(board, agent.color), True, [])
            total_reward += reward
            outcome = 'draw'
            break

        opp_move = get_stockfish_move(stockfish, board, opponent_color, last_move)
        if opp_move is None:
            opp_legal = get_legal_moves(board, opponent_color, last_move)
            if not opp_legal:
                break
            opp_move = random.choice(opp_legal)
        ofrom, oto = opp_move
        opiece = board.grid[ofrom[0]][ofrom[1]]
        oresult = opiece.move(ofrom, oto, board, last_move)
        if oresult == 'promotion':
            board.promote(oto[0], oto[1], Queen(opponent_color))
        last_move = (board.grid[oto[0]][oto[1]], ofrom, oto)

        agent_in_check = is_in_check(board, agent.color, last_move)
        agent_mated = agent_in_check and is_in_checkmate(board, agent.color, last_move)

        material_after = material_balance(board, agent.color)
        eval_after = get_stockfish_eval(stockfish, board, agent.color, last_move) if use_eval else None

        step_reward = agent.compute_reward(
            opp_in_check, agent_in_check, material_before, material_after, eval_before, eval_after
        )
        next_tensor = board_to_tensor(board, agent.color)

        if agent_mated:
            step_reward += agent.loss_reward
            agent.memory.push(state_tensor, action_idx, step_reward, next_tensor, True, [])
            total_reward += step_reward
            outcome = 'loss-checkmated'
            break

        agent_stalemate = (not agent_in_check) and is_stalemate(board, agent.color, last_move)
        if agent_stalemate:
            step_reward += agent.draw_reward
            agent.memory.push(state_tensor, action_idx, step_reward, next_tensor, True, [])
            total_reward += step_reward
            outcome = 'draw'
            break

        next_legal = get_legal_moves(board, agent.color, last_move)
        next_legal_actions = [encode_action(fp, tp) for fp, tp in next_legal]
        agent.memory.push(state_tensor, action_idx, step_reward, next_tensor, False, next_legal_actions)
        total_reward += step_reward
        state_tensor = next_tensor

    board.display()
    return outcome, total_reward


# ─────────────────────────────────────────────────────────────────────────
# Training manager -- runs episodes on a background thread and tracks
# stats, matching the endpoints the Bot page in index.html already calls.
# ─────────────────────────────────────────────────────────────────────────

class TrainingManager:
    def __init__(self):
        self.agent = None
        self.stockfish_path = r"C:\Users\eman2\Documents\GitHub\Project\Game\stockfish\stockfish-windows-x86-64-avx2.exe"
        self.stockfish_depth = 15
        self._thread = None
        self.running = False
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()
        self.wins = 0
        self.games = 0
        self.reward_history = deque(maxlen=100)
        self.stats = {'episode': 0, 'win_rate': 0.0, 'avg_reward': 0.0, 'epsilon': 1.0, 'done': False}

    def configure(self, color='W', lr=1e-3, gamma=0.99, epsilon=1.0, batch_size=64,
                  stockfish_path=None, stockfish_depth=8):
        self.agent = ChessQLearningAgent(color=color, lr=lr, gamma=gamma,
                                          epsilon=epsilon, batch_size=batch_size)
        self.stockfish_path = stockfish_path
        self.stockfish_depth = stockfish_depth
        self.wins = 0
        self.games = 0
        self.reward_history.clear()
        self.stats = {'episode': 0, 'win_rate': 0.0, 'avg_reward': 0.0,
                       'epsilon': self.agent.epsilon, 'done': False}

    def load_agent(self, path):
        if self.agent is None:
            self.configure()
        self.agent.load(path)

    def save_agent(self, path):
        if self.agent is None:
            raise RuntimeError("No agent configured")
        self.agent.save(path)

    def start(self, episodes, save_path=None, save_interval=100):
        if self.agent is None:
            self.configure()
        if self.running:
            return False
        self._stop_flag.clear()
        self.running = True
        self._thread = threading.Thread(
            target=self._run, args=(episodes, save_path, save_interval), daemon=True
        )
        self._thread.start()
        return True

    def _run(self, episodes, save_path, save_interval):
        sf = None
        if self.stockfish_path and Stockfish is not None:
            try:
                sf = Stockfish(path=self.stockfish_path)
                sf.set_depth(self.stockfish_depth)
            except Exception:
                sf = None

        opponent_color = other_color(self.agent.color)

        for ep in range(1, episodes + 1):
            if self._stop_flag.is_set():
                break

            outcome, total_reward = run_episode(self.agent, sf, opponent_color, use_eval=(sf is not None))

            for _ in range(4):
                self.agent.train_step()
            self.agent.decay_epsilon()
            if ep % 50 == 0:
                self.agent.update_target_network()

            with self._lock:
                self.games += 1
                if outcome == 'win':
                    self.wins += 1
                self.reward_history.append(total_reward)
                self.stats['episode'] = ep
                self.stats['win_rate'] = self.wins / self.games
                self.stats['avg_reward'] = sum(self.reward_history) / len(self.reward_history)
                self.stats['epsilon'] = self.agent.epsilon

            if save_path and ep % save_interval == 0:
                self.agent.save(save_path)

        if save_path:
            self.agent.save(save_path)

        with self._lock:
            self.stats['done'] = True
        self.running = False

    def stop(self):
        self._stop_flag.set()

    def get_stats(self):
        with self._lock:
            return dict(self.stats)


# ─────────────────────────────────────────────────────────────────────────
# Standalone smoke test: `python rllearn.py`
# Runs a handful of episodes without Stockfish (random-move opponent,
# material/check reward terms only) just to validate the pipeline.
# ─────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    save_path = "saved_model.pth"
    agent = ChessQLearningAgent(color='W', epsilon=1.0, epsilon_min=0.1,
                                 epsilon_decay=0.9, batch_size=16)
    if os.path.exists(save_path):
        print(f"Loading model from {save_path}...")
        agent.load(save_path)
    else:
        print("No saved model found. Starting fresh training...")
        
    sf = Stockfish(path=r"C:\Users\eman2\Documents\GitHub\Project\Game\stockfish\stockfish-windows-x86-64-avx2.exe")
    sf.set_depth(15)
    for ep in range(1, 5):
        outcome, reward = run_episode(agent, stockfish=sf, opponent_color='B',
                                       max_plies=40, use_eval=False)
        loss = agent.train_step()
        agent.decay_epsilon()
        print(f"episode {ep}: outcome={outcome} reward={reward:.2f} "
              f"epsilon={agent.epsilon:.3f} loss={loss}")

    print(f"Saving model to {save_path}...")
    agent.save(save_path)
    print("Model saved successfully.")

#Glitches - king disappears/gets captured - should be illegal