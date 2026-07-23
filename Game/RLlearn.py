"""
rllearn.py  —  Staged chess RL training with supervised pretraining.

Architecture: ~199M parameters
  - 19-plane board encoding (pieces + castling rights + en-passant)
  - SE-ResNet trunk (Squeeze-and-Excitation residual blocks)
  - Dueling DQN head (separate value + advantage streams)
  - channels=512, num_res=36

Pipeline
--------
  Stage 0 — Opening pretraining  (imitation learning on Lichess openings)
  Stage 1 — Puzzle pretraining   (imitation learning on Lichess tactics puzzles)
  Stage 2 — RL self-play         (off-policy DQN vs. Stockfish)

Usage
-----
  python rllearn.py                      # all three stages
  python rllearn.py --stage openings
  python rllearn.py --stage puzzles
  python rllearn.py --stage rl
  python rllearn.py --stage rl --resume  # resume from checkpoint

Bug fixes vs previous version
------------------------------
  - Stockfish process auto-restarts every 200 episodes + on failure detection
  - board_to_tensor now includes en-passant and castling planes so the network
    sees full positional context; passed as last_move throughout run_episode
  - Illegal-move guard in run_episode: opponent's Stockfish move is validated
    against get_legal_moves before being applied; falls back to legal random
    move rather than crashing or silently corrupting board state
  - Agent instantiation at bottom now uses the large architecture defaults
    (channels=512, num_res=36) so parameter count matches QNetwork defaults
"""
#gpu utilization between 0-2%
import os
import copy
import random
import threading
import sys
import signal
import argparse
from collections import deque
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from pieces import Pawn, Knight, Bishop, Rook, Queen, King, InBounds
from board import Board, is_in_check, is_in_checkmate, is_stalemate

try:
    from stockfish import Stockfish as _SF
except ImportError:
    _SF = None

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PIECE_TYPES  = ['Pawn', 'Knight', 'Bishop', 'Rook', 'Queen', 'King']
PIECE_VALUES = {'Pawn': 1, 'Knight': 3, 'Bishop': 3, 'Rook': 5, 'Queen': 9, 'King': 0}
NUM_ACTIONS  = 64 * 64   # (from_sq, to_sq) — promotions auto-queen
IN_CHANNELS  = 19        # board encoding planes

_PIECE_CLS = {
    'P': (Pawn,   'W'), 'N': (Knight, 'W'), 'B': (Bishop, 'W'),
    'R': (Rook,   'W'), 'Q': (Queen,  'W'), 'K': (King,   'W'),
    'p': (Pawn,   'B'), 'n': (Knight, 'B'), 'b': (Bishop, 'B'),
    'r': (Rook,   'B'), 'q': (Queen,  'B'), 'k': (King,   'B'),
}
_PROMO_CLS = {'q': Queen, 'r': Rook, 'b': Bishop, 'n': Knight}

def other_color(c): return 'B' if c == 'W' else 'W'

SF_PATH = r"C:\Users\eman2\Documents\GitHub\Project\Game\stockfish\stockfish-windows-x86-64-avx2.exe"

# ─────────────────────────────────────────────────────────────────────────────
# Board → tensor  (19 planes)
#
# Plane layout:
#   0– 5  white pieces  (Pawn, Knight, Bishop, Rook, Queen, King)
#   6–11  black pieces
#   12    side to move  (1 = White, 0 = Black)
#   13    white can castle kingside
#   14    white can castle queenside
#   15    black can castle kingside
#   16    black can castle queenside
#   17    en-passant column  (entire column = 1 where EP capture is possible)
#   18    en-passant row     (entire row    = 1 where EP capture is possible)
# ─────────────────────────────────────────────────────────────────────────────

def board_to_tensor(board, side_to_move, last_move=None):
    t = torch.zeros(IN_CHANNELS, 8, 8, dtype=torch.float32)

    # Piece planes
    for r in range(8):
        for c in range(8):
            p = board.grid[r][c]
            if p == ' ': continue
            idx = PIECE_TYPES.index(p.name) + (0 if p.color == 'W' else 6)
            t[idx, r, c] = 1.0

    # Side to move
    t[12] = 1.0 if side_to_move == 'W' else 0.0

    # Castling rights — inferred from piece history
    wk = board.grid[0][4]
    if wk != ' ' and wk.name == 'King' and wk.color == 'W' and not wk.history:
        wr = board.grid[0][7]
        wl = board.grid[0][0]
        if wr != ' ' and wr.name == 'Rook' and wr.color == 'W' and not wr.history:
            t[13] = 1.0
        if wl != ' ' and wl.name == 'Rook' and wl.color == 'W' and not wl.history:
            t[14] = 1.0
    bk = board.grid[7][4]
    if bk != ' ' and bk.name == 'King' and bk.color == 'B' and not bk.history:
        br = board.grid[7][7]
        bl = board.grid[7][0]
        if br != ' ' and br.name == 'Rook' and br.color == 'B' and not br.history:
            t[15] = 1.0
        if bl != ' ' and bl.name == 'Rook' and bl.color == 'B' and not bl.history:
            t[16] = 1.0

    # En-passant target square (from last move)
    if last_move is not None:
        try:
            lmp, (lfr, lfc), (ltr, ltc) = last_move
            if lmp.name == 'Pawn' and abs(lfr - ltr) == 2:
                ep_col = ltc
                ep_row = (lfr + ltr) // 2
                t[17, :, ep_col] = 1.0
                t[18, ep_row, :] = 1.0
        except Exception:
            pass

    return t


def encode_action(fp, tp):
    return (fp[0] * 8 + fp[1]) * 64 + (tp[0] * 8 + tp[1])


def material_balance(board, color):
    own = opp = 0
    for r in range(8):
        for c in range(8):
            p = board.grid[r][c]
            if p == ' ': continue
            v = PIECE_VALUES.get(p.name, 0)
            if p.color == color: own += v
            else: opp += v
    return own - opp


# ─────────────────────────────────────────────────────────────────────────────
# Legal-move generation
# ─────────────────────────────────────────────────────────────────────────────

def _pawn_cands(r, c, col):
    d  = 1 if col == 'W' else -1
    sr = 1 if col == 'W' else 6
    cands = [(r+d, c), (r+d, c-1), (r+d, c+1)]
    if r == sr: cands.append((r + 2*d, c))
    return [(rr, cc) for rr, cc in cands if InBounds(rr, cc)]

def _knight_cands(r, c):
    return [(r+dr, c+dc) for dr, dc in
            [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
            if InBounds(r+dr, c+dc)]

def _slide(r, c, dirs):
    out = []
    for dr, dc in dirs:
        for i in range(1, 8):
            rr, cc = r+dr*i, c+dc*i
            if not InBounds(rr, cc): break
            out.append((rr, cc))
    return out

def _bishop_cands(r, c): return _slide(r, c, [(-1,-1),(-1,1),(1,-1),(1,1)])
def _rook_cands(r, c):   return _slide(r, c, [(-1,0),(1,0),(0,-1),(0,1)])
def _queen_cands(r, c):  return _bishop_cands(r, c) + _rook_cands(r, c)
def _king_cands(r, c):
    out = [(r+dr, c+dc) for dr in(-1,0,1) for dc in(-1,0,1)
           if (dr or dc) and InBounds(r+dr, c+dc)]
    out += [(r, c+2), (r, c-2)]
    return [p for p in out if InBounds(*p)]

_CANDS = {
    'Pawn':   _pawn_cands,
    'Knight': lambda r, c, col: _knight_cands(r, c),
    'Bishop': lambda r, c, col: _bishop_cands(r, c),
    'Rook':   lambda r, c, col: _rook_cands(r, c),
    'Queen':  lambda r, c, col: _queen_cands(r, c),
    'King':   lambda r, c, col: _king_cands(r, c),
}


def get_legal_moves(board, color, last_move=None):
    legal = []
    for r in range(8):
        for c in range(8):
            p = board.grid[r][c]
            if p == ' ' or p.color != color: continue
            for tr, tc in _CANDS[p.name](r, c, color):
                try:
                    tb = copy.deepcopy(board)
                    tb.grid[r][c].move((r, c), (tr, tc), tb, last_move)
                except Exception:
                    continue
                if not is_in_check(tb, color, last_move):
                    legal.append(((r, c), (tr, tc)))
    return legal


def fresh_board():
    b = Board()
    b.grid[0][0]=Rook("W");   b.grid[0][1]=Knight("W"); b.grid[0][2]=Bishop("W")
    b.grid[0][3]=Queen("W");  b.grid[0][4]=King("W");   b.grid[0][5]=Bishop("W")
    b.grid[0][6]=Knight("W"); b.grid[0][7]=Rook("W")
    for i in range(8): b.grid[1][i] = Pawn("W")
    b.grid[7][0]=Rook("B");   b.grid[7][1]=Knight("B"); b.grid[7][2]=Bishop("B")
    b.grid[7][3]=Queen("B");  b.grid[7][4]=King("B");   b.grid[7][5]=Bishop("B")
    b.grid[7][6]=Knight("B"); b.grid[7][7]=Rook("B")
    for i in range(8): b.grid[6][i] = Pawn("B")
    return b


# ─────────────────────────────────────────────────────────────────────────────
# FEN / UCI utilities
# ─────────────────────────────────────────────────────────────────────────────

_DUMMY_HIST = ['moved']

def fen_to_board(fen):
    parts     = fen.strip().split()
    placement = parts[0]
    stm       = 'W' if parts[1] == 'w' else 'B'
    castling  = parts[2] if len(parts) > 2 else '-'
    ep_sq     = parts[3] if len(parts) > 3 else '-'

    board = Board()
    rank, file = 7, 0
    for ch in placement:
        if ch == '/':
            rank -= 1; file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            cls, color = _PIECE_CLS[ch]
            board.grid[rank][file] = cls(color)
            file += 1

    wk = board.grid[0][4]
    if isinstance(wk, King) and wk.color == 'W':
        if 'K' not in castling and 'Q' not in castling:
            wk.history = _DUMMY_HIST[:]
    wr = board.grid[0][7]
    if isinstance(wr, Rook) and wr.color == 'W' and 'K' not in castling:
        wr.history = _DUMMY_HIST[:]
    wl = board.grid[0][0]
    if isinstance(wl, Rook) and wl.color == 'W' and 'Q' not in castling:
        wl.history = _DUMMY_HIST[:]
    bk = board.grid[7][4]
    if isinstance(bk, King) and bk.color == 'B':
        if 'k' not in castling and 'q' not in castling:
            bk.history = _DUMMY_HIST[:]
    br = board.grid[7][7]
    if isinstance(br, Rook) and br.color == 'B' and 'k' not in castling:
        br.history = _DUMMY_HIST[:]
    bl = board.grid[7][0]
    if isinstance(bl, Rook) and bl.color == 'B' and 'q' not in castling:
        bl.history = _DUMMY_HIST[:]

    last_move = None
    if ep_sq != '-':
        ec = ord(ep_sq[0]) - ord('a')
        er = int(ep_sq[1]) - 1
        pawn_row  = er - 1 if stm == 'W' else er + 1
        pawn_from = er + 1 if stm == 'W' else er - 1
        pawn = board.grid[pawn_row][ec]
        if isinstance(pawn, Pawn):
            last_move = (pawn, (pawn_from, ec), (pawn_row, ec))

    return board, stm, last_move


def _parse_uci(uci):
    fc = ord(uci[0]) - ord('a');  fr = int(uci[1]) - 1
    tc = ord(uci[2]) - ord('a');  tr = int(uci[3]) - 1
    promo = uci[4].lower() if len(uci) > 4 else None
    return (fr, fc), (tr, tc), promo


def _apply_uci(board, uci, last_move=None):
    (fr, fc), (tr, tc), promo = _parse_uci(uci)
    piece = board.grid[fr][fc]
    if piece == ' ': return last_move
    try:
        result = piece.move((fr, fc), (tr, tc), board, last_move)
    except Exception:
        board.grid[tr][tc] = board.grid[fr][fc]
        board.grid[fr][fc] = ' '
        result = 'promotion' if promo else None
    if result == 'promotion' or promo:
        board.grid[tr][tc] = _PROMO_CLS.get(promo, Queen)(piece.color)
    return (board.grid[tr][tc], (fr, fc), (tr, tc))


# ─────────────────────────────────────────────────────────────────────────────
# Stockfish helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_stockfish(sf_path=SF_PATH, sf_depth=15):
    """Create a fresh Stockfish process. Returns None if unavailable."""
    if _SF is None or not sf_path: return None
    try:
        sf = _SF(path=sf_path)
        sf.set_depth(sf_depth)
        return sf
    except Exception as e:
        print(f"[stockfish] Failed to start: {e}")
        return None


def local_board_to_fen(board, stm, last_move=None):
    rows = []
    for r in range(7, -1, -1):
        s, empty = '', 0
        for c in range(8):
            p = board.grid[r][c]
            if p == ' ':
                empty += 1
            else:
                if empty: s += str(empty); empty = 0
                s += p.fen
        if empty: s += str(empty)
        rows.append(s)
    placement = '/'.join(rows)
    stm_ch = 'w' if stm == 'W' else 'b'

    castling = ''
    wk = board.grid[0][4]
    if wk != ' ' and wk.name == 'King' and wk.color == 'W' and not wk.history:
        wr, wl = board.grid[0][7], board.grid[0][0]
        if wr != ' ' and wr.name == 'Rook' and wr.color == 'W' and not wr.history: castling += 'K'
        if wl != ' ' and wl.name == 'Rook' and wl.color == 'W' and not wl.history: castling += 'Q'
    bk = board.grid[7][4]
    if bk != ' ' and bk.name == 'King' and bk.color == 'B' and not bk.history:
        br, bl = board.grid[7][7], board.grid[7][0]
        if br != ' ' and br.name == 'Rook' and br.color == 'B' and not br.history: castling += 'k'
        if bl != ' ' and bl.name == 'Rook' and bl.color == 'B' and not bl.history: castling += 'q'
    if not castling: castling = '-'

    ep = '-'
    if last_move:
        try:
            lmp, (lfr, lfc), (ltr, _) = last_move
            if lmp.name == 'Pawn' and abs(lfr - ltr) == 2:
                ep = f"{chr(ord('a') + lfc)}{(lfr + ltr) // 2 + 1}"
        except Exception:
            pass

    return f"{placement} {stm_ch} {castling} {ep} 0 1"


def sf_move(sf, board, color, last_move=None):
    """
    Get Stockfish's best move. Returns ((fr,fc),(tr,tc)) or None.
    Also validates the returned move against our legal-move generator —
    this prevents board corruption if the FEN/UCI mapping ever diverges.
    """
    if sf is None: return None
    try:
        sf.set_fen_position(local_board_to_fen(board, color, last_move),
                            do_validation=False)
        uci = sf.get_best_move()
        if not uci: return None
        (fr, fc), (tr, tc), _ = _parse_uci(uci)
        move = (fr, fc), (tr, tc)

        # Validate: must appear in our legal-move list
        legal = get_legal_moves(board, color, last_move)
        if move not in legal:
            # Move is invalid in our engine — fall back to random legal move
            return random.choice(legal) if legal else None
        return move
    except Exception:
        return None


def sf_eval(sf, board, color, last_move=None):
    if sf is None: return None
    try:
        sf.set_fen_position(local_board_to_fen(board, color, last_move),
                            do_validation=False)
        r = sf.get_evaluation()
        return float(r['value']) if r['type'] == 'cp' \
            else (10000.0 if r['value'] > 0 else -10000.0)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Neural network  —  SE-ResNet trunk + Dueling DQN heads (~199M parameters)
#
# Plane input  : (batch, 19, 8, 8)
# Q-value out  : (batch, 4096)   — Q(s,a) for all (from_sq, to_sq) pairs
#
# Dueling decomposition:
#   Q(s,a) = V(s) + A(s,a) − mean_{a'} A(s,a')
#
# This separates position evaluation (value stream) from move preference
# (advantage stream), which is especially useful in chess where many
# positions are won/lost regardless of which legal move is chosen.
# ─────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, ch, ratio=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(ch, ch // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(ch // ratio, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s = self.pool(x).view(x.size(0), -1)
        s = self.fc(s).view(x.size(0), -1, 1, 1)
        return x * s


class SEResBlock(nn.Module):
    """Residual block with SE channel attention."""
    def __init__(self, ch, se_ratio=16):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)
        self.se    = SEBlock(ch, se_ratio)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.se(self.bn2(self.conv2(x)))
        return self.relu(x + residual)


class QNetwork(nn.Module):
    def __init__(self, num_actions=NUM_ACTIONS, in_channels=IN_CHANNELS,
                 channels=512, num_res=36, se_ratio=16):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Residual trunk
        self.res = nn.Sequential(
            *[SEResBlock(channels, se_ratio) for _ in range(num_res)]
        )

        # Value stream  V(s) → scalar
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, 512), nn.ReLU(),
            nn.Linear(512, 256),     nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Advantage stream  A(s,a) → NUM_ACTIONS scalars
        self.adv_conv = nn.Sequential(
            nn.Conv2d(channels, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.adv_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 64, 2048), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(2048, num_actions),
        )

    def forward(self, x):
        x = self.res(self.stem(x))
        v = self.value_fc(self.value_conv(x))          # (B, 1)
        a = self.adv_fc(self.adv_conv(x))              # (B, NUM_ACTIONS)
        # Dueling aggregation: Q = V + A − mean(A)
        return v + a - a.mean(dim=1, keepdim=True)


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Prioritized Experience Replay (SumTree-based)
#
# Rare terminal events (wins/checkmates) happen infrequently early in training.
# PER samples transitions proportional to |TD error|, so high-surprise events
# (like an unexpected checkmate) are replayed more often.
# ─────────────────────────────────────────────────────────────────────────────

class SumTree:
    """Binary SumTree for O(log n) priority sampling."""
    def __init__(self, cap):
        self.cap  = cap
        self.tree = [0.0] * (2 * cap - 1)
        self.data = [None] * cap
        self.n    = 0
        self.pos  = 0

    def _propagate(self, idx, delta):
        parent = (idx - 1) // 2
        while True:
            self.tree[parent] += delta
            if parent == 0: break
            parent = (parent - 1) // 2

    def update(self, data_idx, priority):
        tree_idx = data_idx + self.cap - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def add(self, priority, data):
        self.data[self.pos] = data
        self.update(self.pos, priority)
        self.pos = (self.pos + 1) % self.cap
        self.n   = min(self.n + 1, self.cap)

    def get(self, s):
        idx = 0
        while True:
            left, right = 2*idx+1, 2*idx+2
            if left >= len(self.tree): break
            idx = left if s <= self.tree[left] else right
            if idx == right: s -= self.tree[left]
        data_idx = idx - (self.cap - 1)
        return data_idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self): return self.tree[0]
    def __len__(self):  return self.n


class PrioritizedReplayBuffer:
    """
    PER buffer with importance-sampling weights.
    alpha controls prioritization strength (0 = uniform, 1 = full priority).
    beta  controls IS weight correction (0 = no correction, 1 = full).
    """
    def __init__(self, cap=60_000, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_steps=100_000):
        self.tree       = SumTree(cap)
        self.alpha      = alpha
        self.beta       = beta_start
        self.beta_end   = beta_end
        self.beta_delta = (beta_end - beta_start) / max(beta_steps, 1)
        self.max_prio   = 1.0
        self._eps       = 1e-6

    def push(self, *transition):
        self.tree.add(self.max_prio ** self.alpha, transition)

    def sample(self, n):
        if len(self.tree) < n:
            return None
        indices, weights, batch = [], [], []
        total   = self.tree.total
        segment = total / n

        for i in range(n):
            lo = segment * i
            hi = segment * (i + 1)
            s  = random.uniform(lo, hi)
            idx, prio, data = self.tree.get(s)
            if data is None: continue
            prob = (prio + self._eps) / (total + self._eps)
            w    = (len(self.tree) * prob) ** (-self.beta)
            indices.append(idx)
            weights.append(w)
            batch.append(data)

        if not batch: return None
        max_w = max(weights)
        weights = [w / max_w for w in weights]
        self.beta = min(self.beta_end, self.beta + self.beta_delta)
        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            prio = (abs(float(err)) + self._eps) ** self.alpha
            self.max_prio = max(self.max_prio, prio)
            self.tree.update(idx, prio)

    def __len__(self): return len(self.tree)


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class ChessQLearningAgent:
    def __init__(self, color='W', lr=3e-4, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9997,
                 batch_size=256, buf_size=60_000,
                 win_r=20.0, draw_r=2.0, loss_r=-20.0,
                 check_b=0.3, check_p=0.3, mat_w=1.0, eval_w=0.2,
                 channels=512, num_res=36, se_ratio=16, device=None):
        self.color         = color
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.win_r         = win_r
        self.draw_r        = draw_r
        self.loss_r        = loss_r
        self.check_b       = check_b
        self.check_p       = check_p
        self.mat_w         = mat_w
        self.eval_w        = eval_w

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[rllearn] device: {self.device}")

        self.policy = QNetwork(channels=channels, num_res=num_res, se_ratio=se_ratio).to(self.device)
        self.target = QNetwork(channels=channels, num_res=num_res, se_ratio=se_ratio).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        n_params = count_parameters(self.policy)
        print(f"[rllearn] parameters: {n_params:,}  ({n_params/1e6:.1f}M)")

        self.opt   = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=2000, eta_min=1e-6)
        self.mem   = PrioritizedReplayBuffer(buf_size)

    def select_action(self, board, legal, last_move=None, training=True):
        if not legal: return None
        if training and random.random() < self.epsilon:
            fp, tp = random.choice(legal)
        else:
            s = board_to_tensor(board, self.color, last_move).unsqueeze(0).to(self.device)
            with torch.no_grad():
                qs = self.policy(s).squeeze(0)
            fp, tp = max(legal, key=lambda m: qs[encode_action(*m)].item())
        return fp, tp, encode_action(fp, tp)

    def step_reward(self, opp_check, self_check, mat_b, mat_a, ev_b, ev_a):
        r  = self.check_b * opp_check - self.check_p * self_check
        r += self.mat_w * (mat_a - mat_b)
        if ev_b is not None and ev_a is not None:
            r += self.eval_w * max(-2.0, min(2.0, (ev_a - ev_b) / 100.0))
        return r

    def train_step(self):
        result = self.mem.sample(self.batch_size)
        if result is None: return None
        batch, indices, weights = result

        S, A, R, S2, D, LA = zip(*batch)
        S   = torch.stack(S).to(self.device)
        A   = torch.tensor(A,  dtype=torch.long,    device=self.device)
        R   = torch.tensor(R,  dtype=torch.float32, device=self.device)
        S2  = torch.stack(S2).to(self.device)
        D   = torch.tensor(D,  dtype=torch.float32, device=self.device)
        W   = torch.tensor(weights, dtype=torch.float32, device=self.device)

        q = self.policy(S).gather(1, A.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: use policy net to select action, target net to evaluate
            next_q_policy = self.policy(S2)
            next_q_target = self.target(S2)
            mx = torch.zeros(len(batch), device=self.device)
            for i, la in enumerate(LA):
                if la and D[i].item() == 0:
                    la_t = torch.tensor(la, dtype=torch.long, device=self.device)
                    best_a = next_q_policy[i, la_t].argmax()
                    mx[i]  = next_q_target[i, la_t[best_a]]
            tgt = R + self.gamma * mx * (1 - D)

        td_errors = (q - tgt).detach().cpu().tolist()
        self.mem.update_priorities(indices, td_errors)

        loss = (W * nn.functional.huber_loss(q, tgt, reduction='none')).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()
        return loss.item()

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def soft_update_target(self, tau=0.005):
        """Soft target update: θ_target ← τ·θ_policy + (1-τ)·θ_target"""
        for tp, pp in zip(self.target.parameters(), self.policy.parameters()):
            tp.data.copy_(tau * pp.data + (1 - tau) * tp.data)

    def decay_eps(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        torch.save({
            'policy':  self.policy.state_dict(),
            'target':  self.target.state_dict(),
            'opt':     self.opt.state_dict(),
            'epsilon': self.epsilon,
            'color':   self.color,
        }, path)
        print(f"[rllearn] saved → {path}")

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck['policy'])
        self.target.load_state_dict(ck['target'])
        if 'opt' in ck:
            try: self.opt.load_state_dict(ck['opt'])
            except Exception: pass
        self.epsilon = ck.get('epsilon', self.epsilon)
        self.color   = ck.get('color',   self.color)
        print(f"[rllearn] loaded ← {path}  (ε={self.epsilon:.4f}, color={self.color})")


# ─────────────────────────────────────────────────────────────────────────────
# Pretraining loss
# ─────────────────────────────────────────────────────────────────────────────

def _imitation_loss(policy, state_t, demo_action, legal_enc, device, margin=0.8):
    if not legal_enc: return None
    q      = policy(state_t.unsqueeze(0).to(device)).squeeze(0)
    q_demo = q[demo_action]
    la_t   = torch.tensor(legal_enc, dtype=torch.long, device=device)
    q_la   = q[la_t]
    margins = torch.where(
        la_t == demo_action,
        torch.zeros(len(la_t), device=device),
        torch.full((len(la_t),), margin, device=device),
    )
    return torch.clamp(q_la + margins - q_demo, min=0).mean()


def _opt_step(opt, losses, policy, clip=1.0):
    if not losses: return
    total = torch.stack(losses).mean()
    opt.zero_grad(); total.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), clip)
    opt.step()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 — Opening pretraining
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_on_openings(agent, save_path=None, epochs=5,
                          batch_size=128, lr=1e-3, margin=0.8,
                          agent_color=None):
    try:
        from datasets import load_dataset
        print("[stage 0] Loading chess-openings dataset …")
        ds   = load_dataset("Lichess/chess-openings", split="train", num_proc=8)
        rows = list(ds)
    except Exception as e:
        print(f"[stage 0] Could not load dataset: {e}")
        return

    opt = optim.AdamW(agent.policy.parameters(), lr=lr, weight_decay=1e-4)
    agent.policy.train()
    ac = agent_color
    print(f"[stage 0] {len(rows)} openings × {epochs} epochs, batch={batch_size}")

    for epoch in range(epochs):
        random.shuffle(rows)
        n_samples = 0; n_skipped = 0; batch_losses = []

        with tqdm(total=len(rows), desc=f"Epoch {epoch+1}/{epochs}", unit="opening") as pbar:
            for row in rows:
                uci_moves = row['uci'].split()
                if not uci_moves: pbar.update(1); continue

                board     = fresh_board()
                last_move = None
                stm       = 'W'

                for uci in uci_moves:
                    if ac is None or stm == ac:
                        try:
                            (fr, fc), (tr, tc), _ = _parse_uci(uci)
                            demo_action = encode_action((fr, fc), (tr, tc))
                            state_t     = board_to_tensor(board, stm, last_move)
                            legal       = get_legal_moves(board, stm, last_move)
                            legal_enc   = [encode_action(fp, tp) for fp, tp in legal]

                            if demo_action in legal_enc:
                                loss = _imitation_loss(agent.policy, state_t,
                                                       demo_action, legal_enc,
                                                       agent.device, margin)
                                if loss is not None:
                                    batch_losses.append(loss)
                                    n_samples += 1
                            else:
                                n_skipped += 1

                            if len(batch_losses) >= batch_size:
                                _opt_step(opt, batch_losses, agent.policy)
                                batch_losses = []
                        except Exception:
                            n_skipped += 1

                    try:
                        last_move = _apply_uci(board, uci, last_move)
                    except Exception:
                        break
                    stm = other_color(stm)

                pbar.update(1)

        _opt_step(opt, batch_losses, agent.policy)
        print(f"[stage 0] epoch {epoch+1}/{epochs}  samples={n_samples}  skipped={n_skipped}")

    agent.policy.eval()
    if save_path: agent.save(save_path)
    print("[stage 0] Opening pretraining complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Puzzle pretraining
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_on_puzzles(agent, save_path=None,
                         max_puzzles=500_000,
                         min_rating=800, max_rating=2400,
                         themes_filter=None,
                         batch_size=128, lr=5e-4, margin=0.8,
                         log_every=10_000):
    try:
        from datasets import load_dataset
        print("[stage 1] Loading chess-puzzles dataset (streaming) …")
        ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True, num_proc=8)
    except Exception as e:
        print(f"[stage 1] Could not load dataset: {e}")
        return

    opt = optim.AdamW(agent.policy.parameters(), lr=lr, weight_decay=1e-4)
    agent.policy.train()

    n_puzzles = 0; n_samples = 0; n_skipped = 0
    batch_losses = []

    print(f"[stage 1] rating {min_rating}–{max_rating}, max={max_puzzles}, themes={themes_filter}")

    with tqdm(total=max_puzzles, desc="Puzzle pretraining", unit="puzzle") as pbar:
        for row in ds:
            rating = row.get('Rating', 0)
            if not (min_rating <= rating <= max_rating): continue

            if themes_filter is not None:
                themes = row.get('Themes', [])
                if isinstance(themes, str): themes = themes.split()
                if not any(t in themes for t in themes_filter): continue

            if max_puzzles and n_puzzles >= max_puzzles: break

            moves = row['Moves'].split()
            if len(moves) < 2: continue

            try:
                board, fen_stm, last_move = fen_to_board(row['FEN'])
            except Exception:
                continue

            solver_color = other_color(fen_stm)
            n_puzzles += 1

            try:
                last_move = _apply_uci(board, moves[0], last_move)
            except Exception:
                pbar.update(1); continue

            for i in range(1, len(moves), 2):
                demo_uci = moves[i]
                try:
                    (fr, fc), (tr, tc), _ = _parse_uci(demo_uci)
                    demo_action = encode_action((fr, fc), (tr, tc))
                    state_t     = board_to_tensor(board, solver_color, last_move)
                    legal       = get_legal_moves(board, solver_color, last_move)
                    legal_enc   = [encode_action(fp, tp) for fp, tp in legal]

                    if not legal or demo_action not in legal_enc:
                        n_skipped += 1; break

                    loss = _imitation_loss(agent.policy, state_t,
                                           demo_action, legal_enc,
                                           agent.device, margin)
                    if loss is not None:
                        batch_losses.append(loss); n_samples += 1

                    last_move = _apply_uci(board, demo_uci, last_move)
                    if i + 1 < len(moves):
                        last_move = _apply_uci(board, moves[i+1], last_move)

                    if len(batch_losses) >= batch_size:
                        _opt_step(opt, batch_losses, agent.policy)
                        batch_losses = []
                except Exception:
                    n_skipped += 1; break

            pbar.update(1)

            if n_puzzles % log_every == 0:
                print(f"[stage 1] puzzles={n_puzzles}  samples={n_samples}  skipped={n_skipped}")
                if save_path:
                    agent.policy.eval()
                    agent.save(save_path.replace('.pth', f'_p{n_puzzles}.pth'))
                    agent.policy.train()

    _opt_step(opt, batch_losses, agent.policy)
    agent.policy.eval()
    print(f"[stage 1] Complete: puzzles={n_puzzles}  samples={n_samples}  skipped={n_skipped}")
    if save_path: agent.save(save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — RL episode
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(agent, stockfish, opp_color, max_plies=150,
                use_eval=True, learn_from_opponent=True):
    """
    One training game. Agent plays agent.color; stockfish plays opp_color.

    Stockfish validation fix: sf_move() now cross-checks the returned UCI
    move against our own get_legal_moves() before applying it. If Stockfish's
    process has drifted or crashed, the divergent move is caught here rather
    than corrupting the board state and feeding the agent false learning signal.

    Returns (outcome, total_reward, sf_was_available).
    """
    board     = fresh_board()
    last_move = None
    s_t       = board_to_tensor(board, agent.color, last_move)
    total_r   = 0.0
    outcome   = 'draw'
    sf_used   = False

    for _ in range(max_plies):
        # ── Agent's turn ──────────────────────────────────────────────────────
        legal = get_legal_moves(board, agent.color, last_move)
        if not legal:
            outcome = 'loss-no-moves' if is_in_check(board, agent.color, last_move) else 'draw'
            break

        mat_b = material_balance(board, agent.color)
        ev_b  = sf_eval(stockfish, board, agent.color, last_move) if use_eval else None

        result = agent.select_action(board, legal, last_move, training=True)
        if result is None: break
        fp, tp, act = result

        piece = board.grid[fp[0]][fp[1]]
        res   = piece.move(fp, tp, board, last_move)
        if res == 'promotion': board.promote(tp[0], tp[1], Queen(agent.color))
        last_move = (board.grid[tp[0]][tp[1]], fp, tp)

        opp_check = is_in_check(board, opp_color, last_move)
        opp_mate  = opp_check and is_in_checkmate(board, opp_color, last_move)
        if opp_mate:
            nt = board_to_tensor(board, agent.color, last_move)
            agent.mem.push(s_t, act, agent.win_r, nt, True, [])
            total_r += agent.win_r; outcome = 'win'; break

        opp_stale = not opp_check and is_stalemate(board, opp_color, last_move)
        if opp_stale:
            nt = board_to_tensor(board, agent.color, last_move)
            agent.mem.push(s_t, act, agent.draw_r, nt, True, [])
            total_r += agent.draw_r; outcome = 'draw'; break

        # ── Opponent's turn ───────────────────────────────────────────────────
        s_opp = board_to_tensor(board, opp_color, last_move) if learn_from_opponent else None

        # sf_move already validates the move against legal moves internally
        opp_mv = sf_move(stockfish, board, opp_color, last_move)
        if opp_mv is not None: sf_used = True

        if opp_mv is None:
            opp_legal = get_legal_moves(board, opp_color, last_move)
            if not opp_legal: break
            opp_mv = random.choice(opp_legal)

        ofrom, oto = opp_mv
        op    = board.grid[ofrom[0]][ofrom[1]]
        if op == ' ':
            # Board state / stockfish mismatch — abort this game cleanly
            break
        ores  = op.move(ofrom, oto, board, last_move)
        o_act = encode_action(ofrom, oto)
        if ores == 'promotion': board.promote(oto[0], oto[1], Queen(opp_color))
        last_move = (board.grid[oto[0]][oto[1]], ofrom, oto)

        self_check = is_in_check(board, agent.color, last_move)
        self_mate  = self_check and is_in_checkmate(board, agent.color, last_move)
        mat_a      = material_balance(board, agent.color)
        ev_a       = sf_eval(stockfish, board, agent.color, last_move) if use_eval else None
        step_r     = agent.step_reward(opp_check, self_check, mat_b, mat_a, ev_b, ev_a)
        next_t     = board_to_tensor(board, agent.color, last_move)

        # Opponent experience (negated reward, opponent's perspective)
        if learn_from_opponent and s_opp is not None:
            nla_opp = get_legal_moves(board, agent.color, last_move)
            nla_enc = [encode_action(f, t) for f, t in nla_opp]
            opp_next = board_to_tensor(board, opp_color, last_move)
            agent.mem.push(s_opp, o_act, -step_r, opp_next, self_mate, nla_enc)

        if self_mate:
            step_r += agent.loss_r
            agent.mem.push(s_t, act, step_r, next_t, True, [])
            total_r += step_r; outcome = 'loss'; break

        self_stale = not self_check and is_stalemate(board, agent.color, last_move)
        if self_stale:
            step_r += agent.draw_r
            agent.mem.push(s_t, act, step_r, next_t, True, [])
            total_r += step_r; outcome = 'draw'; break

        nla     = get_legal_moves(board, agent.color, last_move)
        nla_enc = [encode_action(f, t) for f, t in nla]
        agent.mem.push(s_t, act, step_r, next_t, False, nla_enc)
        total_r += step_r
        s_t = next_t

    return outcome, total_r, sf_used


# ─────────────────────────────────────────────────────────────────────────────
# TrainingManager — called by game.py Flask endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TrainingManager:
    def __init__(self):
        self.agent    = None
        self.sf_path  = SF_PATH
        self.sf_depth = 15
        self._thread  = None
        self.running  = False
        self._stop    = threading.Event()
        self._lock    = threading.Lock()
        self.wins = self.games = 0
        self._rh  = deque(maxlen=100)
        self.stats = {
            'episode': 0, 'win_rate': 0., 'avg_reward': 0.,
            'epsilon': 1., 'done': False, 'sf_active': False,
        }

    def configure(self, color='W', lr=3e-4, gamma=0.99, epsilon=1.0,
                  batch_size=256, sf_path=None, sf_depth=15,
                  channels=512, num_res=36, se_ratio=16):
        self.agent = ChessQLearningAgent(
            color=color, lr=lr, gamma=gamma, epsilon=epsilon,
            batch_size=batch_size, channels=channels,
            num_res=num_res, se_ratio=se_ratio,
        )
        if sf_path: self.sf_path = sf_path
        self.sf_depth = sf_depth
        self.wins = self.games = 0; self._rh.clear()
        self.stats = {
            'episode': 0, 'win_rate': 0., 'avg_reward': 0.,
            'epsilon': self.agent.epsilon, 'done': False, 'sf_active': False,
        }

    def load_agent(self, path):
        if self.agent is None: self.configure()
        self.agent.load(path)

    def save_agent(self, path):
        if self.agent is None: raise RuntimeError("No agent configured")
        self.agent.save(path)

    def start(self, episodes, save_path=None, save_interval=100):
        if self.agent is None: self.configure()
        if self.running: return False
        self._stop.clear(); self.running = True
        self._thread = threading.Thread(
            target=self._run, args=(episodes, save_path, save_interval), daemon=True)
        self._thread.start(); return True

    def _run(self, episodes, save_path, save_interval):
        # Create initial Stockfish process
        sf           = _make_stockfish(self.sf_path, self.sf_depth)
        sf_fail_count = 0
        opp = other_color(self.agent.color)

        for ep in range(1, episodes + 1):
            if self._stop.is_set(): break

            # Restart Stockfish every 200 episodes or after repeated failures
            if ep % 200 == 0 or sf_fail_count >= 5:
                sf = _make_stockfish(self.sf_path, self.sf_depth)
                sf_fail_count = 0

            outcome, total_r, sf_used = run_episode(
                self.agent, sf, opp,
                use_eval=(sf is not None),
                learn_from_opponent=True,
            )

            if sf is not None and not sf_used:
                sf_fail_count += 1

            # Train: 8 gradient steps per episode, soft target update every step
            for _ in range(8):
                self.agent.train_step()
            self.agent.soft_update_target(tau=0.005)
            self.agent.decay_eps()
            if ep % 200 == 0: self.agent.sched.step()

            with self._lock:
                self.games += 1
                if outcome == 'win': self.wins += 1
                self._rh.append(total_r)
                self.stats.update({
                    'episode':    ep,
                    'win_rate':   self.wins / self.games,
                    'avg_reward': sum(self._rh) / len(self._rh),
                    'epsilon':    self.agent.epsilon,
                    'sf_active':  sf is not None and sf_fail_count < 5,
                })

            if save_path and ep % save_interval == 0:
                self.agent.save(save_path)

        if save_path: self.agent.save(save_path)
        with self._lock: self.stats['done'] = True
        self.running = False

    def stop(self):      self._stop.set()
    def get_stats(self):
        with self._lock: return dict(self.stats)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — staged training
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Staged chess RL training (~199M params)')
    parser.add_argument('--stage', choices=['openings', 'puzzles', 'rl', 'all'],
                        default='all')
    parser.add_argument('--resume',         action='store_true')
    parser.add_argument('--color',          default='W', choices=['W', 'B'])
    parser.add_argument('--rl-episodes',    type=int, default=200_000)
    parser.add_argument('--max-puzzles',    type=int, default=500_000)
    parser.add_argument('--opening-epochs', type=int, default=10)
    args = parser.parse_args()

    os.makedirs('models', exist_ok=True)
    PATHS = {
        'openings': 'models/stage0_openings.pth',
        'puzzles':  'models/stage1_puzzles.pth',
        'rl':       'models/chess_bot.pth',
    }

    print("num threads: " + str(torch.get_num_threads()))

    # ── Create agent with large architecture (channels=512, num_res=36 → ~199M) ──
    agent = ChessQLearningAgent(
        color=args.color,
        epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9997,
        batch_size=256,
        channels=512, num_res=36, se_ratio=16,
    )

    # ── Stage 0: Opening pretraining ──────────────────────────────────────────
    if args.stage in ('openings', 'all'):
        if args.resume and os.path.exists(PATHS['openings']):
            agent.load(PATHS['openings'])
        pretrain_on_openings(
            agent, save_path=PATHS['openings'],
            epochs=args.opening_epochs,
            batch_size=128, lr=1e-3, margin=0.8,
            agent_color=None,
        )


    # ── Stage 1: Puzzle pretraining ───────────────────────────────────────────
    if args.stage in ('puzzles', 'all'):
        for ck in (PATHS['openings'],):
            if os.path.exists(ck): agent.load(ck); break
        if args.resume and os.path.exists(PATHS['puzzles']):
            agent.load(PATHS['puzzles'])
        pretrain_on_puzzles(
            agent, save_path=PATHS['puzzles'],
            max_puzzles=args.max_puzzles,
            min_rating=800, max_rating=2400,
            themes_filter=None,
            batch_size=128, lr=5e-4, margin=0.8,
            log_every=10_000,
        )

    # ── Stage 2: RL against Stockfish ─────────────────────────────────────────
    if args.stage in ('rl', 'all'):
        for ck in (PATHS['puzzles'], PATHS['openings']):
            if os.path.exists(ck): agent.load(ck); break
        if args.resume and os.path.exists(PATHS['rl']):
            agent.load(PATHS['rl'])
        if not args.resume:
            agent.epsilon = 0.4   # exploit pretrained knowledge, but still explore

        sf           = _make_stockfish(SF_PATH, 15)
        sf_fail_count = 0
        opp_color    = other_color(agent.color)
        ep           = 0

        def _save_exit(*_):
            print('\n[stage 2] Saving …'); agent.save(PATHS['rl']); sys.exit(0)
        signal.signal(signal.SIGINT, _save_exit)

        print(f"[stage 2] Starting RL: {args.rl_episodes} episodes, "
              f"agent={agent.color}, sf={'yes' if sf else 'no'}")

        while ep < args.rl_episodes:
            # Restart Stockfish every 200 episodes or on repeated failure
            if ep % 200 == 0 or sf_fail_count >= 5:
                sf = _make_stockfish(SF_PATH, 15)
                sf_fail_count = 0
                if sf: print(f"[stage 2] Stockfish restarted at ep {ep}")

            outcome, reward, sf_used = run_episode(
                agent, sf, opp_color,
                use_eval=(sf is not None),
                learn_from_opponent=True,
            )
            if sf is not None and not sf_used:
                sf_fail_count += 1

            for _ in range(8): agent.train_step()
            agent.soft_update_target(tau=0.005)
            agent.decay_eps()
            if ep % 200 == 0: agent.sched.step()

            sf_tag = '✓' if (sf is not None and sf_fail_count < 5) else '✗'
            print(f"ep {ep:6d}: {outcome:<18s} r={reward:+7.2f}  "
                  f"ε={agent.epsilon:.4f}  sf={sf_tag}")

            if ep % 100 == 0:
                agent.save(PATHS['rl'])
                print("── saved ──")
            ep += 1

        agent.save(PATHS['rl'])
        print("[stage 2] RL training complete.")