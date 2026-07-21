"""
rllearn.py  —  Staged chess RL training with supervised pretraining.

Pipeline
--------
  Stage 0 — Opening pretraining  (imitation learning on Lichess openings)
  Stage 1 — Puzzle pretraining   (imitation learning on Lichess tactics puzzles)
  Stage 2 — RL self-play         (off-policy DQN vs. Stockfish)

Checkpoints saved after every stage and periodically within each stage.

Usage
-----
  python rllearn.py                        # run all three stages in order
  python rllearn.py --stage openings       # only stage 0
  python rllearn.py --stage puzzles        # only stage 1
  python rllearn.py --stage rl             # only stage 2
  python rllearn.py --stage rl --resume    # resume RL from latest checkpoint

Prerequisites
-------------
  pip install torch datasets
  (stockfish binary + `pip install stockfish` for stage 2 eval rewards)

Dataset schemas (confirmed from HuggingFace)
---------------------------------------------
  Lichess/chess-openings : eco-volume, eco, name, pgn, uci, epd
  Lichess/chess-puzzles  : PuzzleId, FEN, Moves, Rating, Themes, ...

    Puzzle format (important!):
      FEN   — position BEFORE the first move in Moves
      Moves — space-separated UCI.  Moves[0] is the opponent's "trigger" move
              that sets up the puzzle; Moves[1], Moves[3], ... are the solver's
              correct replies (odd-indexed, 1-based).
      solver_color = other(FEN side-to-move)
"""

##Needs more complex architecture - only has 31mil trainable params

import os
import copy
import random
import threading
import sys
import signal
import argparse
from collections import deque

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
NUM_ACTIONS  = 64 * 64        # (from_sq, to_sq) — promotion auto-queens

_PIECE_CLS = {
    'P': (Pawn,   'W'), 'N': (Knight, 'W'), 'B': (Bishop, 'W'),
    'R': (Rook,   'W'), 'Q': (Queen,  'W'), 'K': (King,   'W'),
    'p': (Pawn,   'B'), 'n': (Knight, 'B'), 'b': (Bishop, 'B'),
    'r': (Rook,   'B'), 'q': (Queen,  'B'), 'k': (King,   'B'),
}
_PROMO_CLS = {'q': Queen, 'r': Rook, 'b': Bishop, 'n': Knight}

def other_color(c): return 'B' if c == 'W' else 'W'


# ─────────────────────────────────────────────────────────────────────────────
# Board → tensor / action encoding
# ─────────────────────────────────────────────────────────────────────────────

def board_to_tensor(board, side_to_move):
    """13 planes: 6 piece-types × 2 colors + 1 side-to-move, each 8×8."""
    t = torch.zeros(13, 8, 8, dtype=torch.float32)
    for r in range(8):
        for c in range(8):
            p = board.grid[r][c]
            if p == ' ':
                continue
            idx = PIECE_TYPES.index(p.name) + (0 if p.color == 'W' else 6)
            t[idx, r, c] = 1.0
    t[12] = 1.0 if side_to_move == 'W' else 0.0
    return t


def encode_action(fp, tp):
    return (fp[0] * 8 + fp[1]) * 64 + (tp[0] * 8 + tp[1])


def material_balance(board, color):
    own = opp = 0
    for r in range(8):
        for c in range(8):
            p = board.grid[r][c]
            if p == ' ':
                continue
            v = PIECE_VALUES.get(p.name, 0)
            if p.color == color:
                own += v
            else:
                opp += v
    return own - opp


# ─────────────────────────────────────────────────────────────────────────────
# Legal-move generation (candidate-first, avoids brute-forcing all 64 targets)
# ─────────────────────────────────────────────────────────────────────────────

def _pawn_cands(r, c, col):
    d  = 1 if col == 'W' else -1
    sr = 1 if col == 'W' else 6
    cands = [(r+d, c), (r+d, c-1), (r+d, c+1)]
    if r == sr:
        cands.append((r + 2*d, c))
    return [(rr, cc) for rr, cc in cands if InBounds(rr, cc)]

def _knight_cands(r, c):
    return [(r+dr, c+dc) for dr, dc in
            [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
            if InBounds(r+dr, c+dc)]

def _slide(r, c, dirs):
    out = []
    for dr, dc in dirs:
        for i in range(1, 8):
            rr, cc = r + dr*i, c + dc*i
            if not InBounds(rr, cc):
                break
            out.append((rr, cc))
    return out

def _bishop_cands(r, c): return _slide(r, c, [(-1,-1),(-1,1),(1,-1),(1,1)])
def _rook_cands(r, c):   return _slide(r, c, [(-1,0),(1,0),(0,-1),(0,1)])
def _queen_cands(r, c):  return _bishop_cands(r, c) + _rook_cands(r, c)
def _king_cands(r, c):
    out = [(r+dr, c+dc) for dr in (-1,0,1) for dc in (-1,0,1)
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
            if p == ' ' or p.color != color:
                continue
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
# FEN / UCI utilities  (needed for pretraining)
# ─────────────────────────────────────────────────────────────────────────────

_DUMMY_HIST = ['moved']   # non-empty history → piece has moved → no castling

def fen_to_board(fen):
    """
    Parse a FEN string into (Board, side_to_move, last_move_or_None).

    Castling rights in the FEN are honoured by marking the relevant King/Rook
    objects with a dummy history so that King.move() won't attempt illegal castling.
    The en-passant square is used to reconstruct `last_move` so that pawn
    capture logic works correctly during legal-move generation.
    """
    parts = fen.strip().split()
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

    # White castling rights
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

    # Black castling rights
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

    # Reconstruct last_move from en-passant square so Pawn.move() is accurate
    last_move = None
    if ep_sq != '-':
        ec = ord(ep_sq[0]) - ord('a')
        er = int(ep_sq[1]) - 1
        # stm='W' → black pawn double-pushed last → pawn is one rank below ep square
        # stm='B' → white pawn double-pushed last → pawn is one rank above ep square
        if stm == 'W':
            pawn_row = er - 1;  pawn_from = er + 1
        else:
            pawn_row = er + 1;  pawn_from = er - 1
        pawn = board.grid[pawn_row][ec]
        if isinstance(pawn, Pawn):
            last_move = (pawn, (pawn_from, ec), (pawn_row, ec))

    return board, stm, last_move


def _parse_uci(uci):
    """'e2e4' → ((1,4), (3,4), None);  'e7e8q' → ((6,4), (7,4), 'q')"""
    fc = ord(uci[0]) - ord('a');  fr = int(uci[1]) - 1
    tc = ord(uci[2]) - ord('a');  tr = int(uci[3]) - 1
    promo = uci[4].lower() if len(uci) > 4 else None
    return (fr, fc), (tr, tc), promo


def _apply_uci(board, uci, last_move=None):
    """
    Apply a UCI move string to `board` in-place.
    Falls back to direct grid manipulation if piece.move() raises (e.g. wrong
    history on a FEN-loaded piece).  Returns the new last_move tuple.
    """
    (fr, fc), (tr, tc), promo = _parse_uci(uci)
    piece = board.grid[fr][fc]
    if piece == ' ':
        return last_move

    try:
        result = piece.move((fr, fc), (tr, tc), board, last_move)
    except Exception:
        # Raw grid move (bypasses validation — only used for known-valid UCI)
        captured = board.grid[tr][tc]
        board.grid[tr][tc] = board.grid[fr][fc]
        board.grid[fr][fc] = ' '
        result = 'promotion' if promo else None

    if result == 'promotion' or promo:
        cls = _PROMO_CLS.get(promo, Queen)
        board.grid[tr][tc] = cls(piece.color)

    moved = board.grid[tr][tc]
    return (moved, (fr, fc), (tr, tc))


# ─────────────────────────────────────────────────────────────────────────────
# Stockfish helpers
# ─────────────────────────────────────────────────────────────────────────────

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
        lmp, (lfr, lfc), (ltr, _) = last_move
        if lmp.name == 'Pawn' and abs(lfr - ltr) == 2:
            ep = f"{chr(ord('a') + lfc)}{(lfr + ltr) // 2 + 1}"

    return f"{placement} {stm_ch} {castling} {ep} 0 1"


def sf_move(sf, board, color, last_move=None):
    if sf is None: return None
    try:
        sf.set_fen_position(local_board_to_fen(board, color, last_move), do_validation=False)
        uci = sf.get_best_move()
        if not uci: return None
        (fr, fc), (tr, tc), _ = _parse_uci(uci)
        return (fr, fc), (tr, tc)
    except: return None


def sf_eval(sf, board, color, last_move=None):
    if sf is None: return None
    try:
        sf.set_fen_position(local_board_to_fen(board, color, last_move), do_validation=False)
        r = sf.get_evaluation()
        return float(r['value']) if r['type'] == 'cp' \
            else (10000.0 if r['value'] > 0 else -10000.0)
    except: return None


# ─────────────────────────────────────────────────────────────────────────────
# Neural network — deep residual Q-network (AlphaZero-style trunk)
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.net(x) + x)


class QNetwork(nn.Module):
    """
    Residual Q-network.
    Input : (batch, 13, 8, 8) board tensor
    Output: (batch, NUM_ACTIONS) Q-values for every (from_sq, to_sq) pair
    """
    def __init__(self, num_actions=NUM_ACTIONS, channels=1024, num_res=30):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(13, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(*[ResBlock(channels) for _ in range(num_res)])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 8 * 8, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, 512),               nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.head(self.res(self.stem(x)))


# ─────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, cap=40_000):
        self.buf = deque(maxlen=cap)
    def push(self, *args):     self.buf.append(args)
    def sample(self, n):       return random.sample(self.buf, n)
    def __len__(self):         return len(self.buf)


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class ChessQLearningAgent:
    def __init__(self, color='W', lr=3e-4, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.997,
                 batch_size=256, buf_size=40_000,
                 win_r=20.0, draw_r=2.0, loss_r=-20.0,
                 check_b=0.3, check_p=0.3, mat_w=1.0, eval_w=0.2,
                 channels=256, num_res=10, device=None):
        self.color          = color
        self.gamma          = gamma
        self.epsilon        = epsilon
        self.epsilon_min    = epsilon_min
        self.epsilon_decay  = epsilon_decay
        self.batch_size     = batch_size
        self.win_r          = win_r
        self.draw_r         = draw_r
        self.loss_r         = loss_r
        self.check_b        = check_b
        self.check_p        = check_p
        self.mat_w          = mat_w
        self.eval_w         = eval_w

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[rllearn] device: {self.device}")
        self.policy = QNetwork(channels=channels, num_res=num_res).to(self.device)
        self.target = QNetwork(channels=channels, num_res=num_res).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.opt   = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=1000, eta_min=1e-5)
        self.mem   = ReplayBuffer(buf_size)

    # ── Action selection ──────────────────────────────────────────────────────
    def select_action(self, board, legal, training=True):
        if not legal: return None
        if training and random.random() < self.epsilon:
            fp, tp = random.choice(legal)
        else:
            s = board_to_tensor(board, self.color).unsqueeze(0).to(self.device)
            with torch.no_grad():
                qs = self.policy(s).squeeze(0)
            fp, tp = max(legal, key=lambda m: qs[encode_action(*m)].item())
        return fp, tp, encode_action(fp, tp)

    # ── Per-step reward shaping ───────────────────────────────────────────────
    def step_reward(self, opp_check, self_check, mat_b, mat_a, ev_b, ev_a):
        r  = self.check_b * opp_check - self.check_p * self_check
        r += self.mat_w * (mat_a - mat_b)
        if ev_b is not None and ev_a is not None:
            r += self.eval_w * max(-2.0, min(2.0, (ev_a - ev_b) / 100.0))
        return r

    # ── Off-policy Q-learning update (Huber loss) ────────────────────────────
    def train_step(self):
        if len(self.mem) < self.batch_size: return None
        S, A, R, S2, D, LA = zip(*self.mem.sample(self.batch_size))
        S  = torch.stack(S).to(self.device)
        A  = torch.tensor(A,  dtype=torch.long,    device=self.device)
        R  = torch.tensor(R,  dtype=torch.float32, device=self.device)
        S2 = torch.stack(S2).to(self.device)
        D  = torch.tensor(D,  dtype=torch.float32, device=self.device)

        q = self.policy(S).gather(1, A.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq = self.target(S2)
            mx = torch.zeros(len(S), device=self.device)
            for i, la in enumerate(LA):
                if la and D[i].item() == 0:
                    idx = torch.tensor(la, dtype=torch.long, device=self.device)
                    mx[i] = nq[i, idx].max()
            tgt = R + self.gamma * mx * (1 - D)

        loss = nn.functional.huber_loss(q, tgt)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()
        return loss.item()

    def update_target(self): self.target.load_state_dict(self.policy.state_dict())
    def decay_eps(self):     self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        torch.save({'policy': self.policy.state_dict(),
                    'target': self.target.state_dict(),
                    'epsilon': self.epsilon,
                    'color':   self.color}, path)
        print(f"[rllearn] saved → {path}")

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck['policy'])
        self.target.load_state_dict(ck['target'])
        self.epsilon = ck.get('epsilon', self.epsilon)
        self.color   = ck.get('color',   self.color)
        print(f"[rllearn] loaded ← {path}  (ε={self.epsilon:.4f}, color={self.color})")


# ─────────────────────────────────────────────────────────────────────────────
# Pretraining utilities
# ─────────────────────────────────────────────────────────────────────────────

def _imitation_loss(policy, state_t, demo_action, legal_enc, device, margin=0.8):
    """
    Large-margin imitation loss (DQfD-style).
    Forces  Q(s, a_demo) ≥ Q(s, a) + margin  for every other legal action a.
    Returns a scalar tensor or None if no legal actions.
    """
    if not legal_enc: return None
    q = policy(state_t.unsqueeze(0).to(device)).squeeze(0)
    q_demo = q[demo_action]
    la_t   = torch.tensor(legal_enc, dtype=torch.long, device=device)
    q_la   = q[la_t]
    margins = torch.where(la_t == demo_action,
                          torch.zeros(len(la_t), device=device),
                          torch.full((len(la_t),), margin, device=device))
    return torch.clamp(q_la + margins - q_demo, min=0).mean()


def _opt_step(opt, losses, policy, clip=1.0):
    """Accumulate a list of losses and do one optimizer step."""
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
    """
    Supervised pretraining on the Lichess chess-openings dataset.

    Dataset: Lichess/chess-openings (≈3 600 openings)
    Field used: 'uci'  — space-separated UCI moves from the start position.

    For each move in each opening line, if it's `agent_color`'s turn (or both
    colors when agent_color=None), we create a training sample that pushes the
    network to prefer that opening move over other legal moves.
    """
    try:
        from datasets import load_dataset
        print("[stage 0] Loading chess-openings dataset …")
        ds = load_dataset("Lichess/chess-openings", split="train")
        rows = list(ds)
    except Exception as e:
        print(f"[stage 0] Could not load openings dataset: {e}")
        print("          Install with: pip install datasets")
        return

    opt = optim.AdamW(agent.policy.parameters(), lr=lr, weight_decay=1e-4)
    agent.policy.train()
    ac = agent_color  # None → train on both colors

    print(f"[stage 0] {len(rows)} openings × {epochs} epochs, batch={batch_size}")

    for epoch in range(epochs):
        random.shuffle(rows)
        n_samples = 0; n_skipped = 0; batch_losses = []

        for row in rows:
            uci_moves = row['uci'].split()
            if not uci_moves: continue

            board     = fresh_board()
            last_move = None
            stm       = 'W'       # white always starts

            for uci in uci_moves:
                if ac is None or stm == ac:
                    try:
                        (fr, fc), (tr, tc), _ = _parse_uci(uci)
                        demo_action = encode_action((fr, fc), (tr, tc))
                        state_t     = board_to_tensor(board, stm)
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

                # Apply move regardless (need board state to advance)
                try:
                    last_move = _apply_uci(board, uci, last_move)
                except Exception:
                    break   # invalid position — stop this opening line
                stm = other_color(stm)

        _opt_step(opt, batch_losses, agent.policy)
        print(f"[stage 0] epoch {epoch+1}/{epochs}  samples={n_samples}  skipped={n_skipped}")

    agent.policy.eval()
    if save_path:
        agent.save(save_path)
    print("[stage 0] Opening pretraining complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Puzzle pretraining
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_on_puzzles(agent, save_path=None,
                         max_puzzles=150_000,
                         min_rating=900, max_rating=2200,
                         themes_filter=None,
                         batch_size=128, lr=5e-4, margin=0.8,
                         log_every=2000):
    """
    Supervised pretraining on the Lichess chess-puzzles dataset.

    Dataset: Lichess/chess-puzzles (~5.8 M puzzles, streamed for memory efficiency)
    Fields:  FEN, Moves, Rating, Themes

    Puzzle format recap:
      Moves[0]           — opponent's trigger move (apply, then it's solver's turn)
      Moves[1,3,5,…]     — solver's correct moves  ← these become training samples
      Moves[2,4,6,…]     — opponent's forced responses (apply, advance board)
      solver_color       = other(FEN side-to-move)

    Parameters
    ----------
    max_puzzles    : cap on number of puzzles to use (None = all)
    min/max_rating : filter puzzles to a tactical difficulty range
    themes_filter  : list of theme strings; keep puzzle if ANY theme matches
                     (e.g. ['mate','pin','fork']). None = keep all themes.
    """
    try:
        from datasets import load_dataset
        print("[stage 1] Loading chess-puzzles dataset (streaming) …")
        ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    except Exception as e:
        print(f"[stage 1] Could not load puzzles dataset: {e}")
        print("          Install with: pip install datasets")
        return

    opt = optim.AdamW(agent.policy.parameters(), lr=lr, weight_decay=1e-4)
    agent.policy.train()

    n_puzzles = 0; n_samples = 0; n_skipped = 0
    batch_losses = []

    print(f"[stage 1] Filtering rating {min_rating}–{max_rating}, "
          f"max_puzzles={max_puzzles}, themes={themes_filter}")

    for row in ds:
        # ── Filter ──────────────────────────────────────────────────────────
        rating = row.get('Rating', 0)
        if not (min_rating <= rating <= max_rating):
            continue

        if themes_filter is not None:
            themes = row.get('Themes', [])
            # Themes field may be a list or a space-separated string
            if isinstance(themes, str): themes = themes.split()
            if not any(t in themes for t in themes_filter):
                continue

        if max_puzzles and n_puzzles >= max_puzzles:
            break

        moves = row['Moves'].split()
        if len(moves) < 2:
            continue   # need trigger + at least one solution move

        # ── Set up board from FEN ────────────────────────────────────────────
        try:
            board, fen_stm, last_move = fen_to_board(row['FEN'])
        except Exception:
            continue

        solver_color = other_color(fen_stm)
        n_puzzles += 1

        # ── Apply trigger move (Moves[0]) ────────────────────────────────────
        try:
            last_move = _apply_uci(board, moves[0], last_move)
        except Exception:
            continue

        # ── Extract solver's moves (odd-indexed: 1, 3, 5, …) ────────────────
        puzzle_ok = True
        for i in range(1, len(moves), 2):
            demo_uci = moves[i]
            try:
                (fr, fc), (tr, tc), _ = _parse_uci(demo_uci)
                demo_action = encode_action((fr, fc), (tr, tc))
                state_t     = board_to_tensor(board, solver_color)
                legal       = get_legal_moves(board, solver_color, last_move)
                legal_enc   = [encode_action(fp, tp) for fp, tp in legal]

                if not legal or demo_action not in legal_enc:
                    n_skipped += 1
                    puzzle_ok = False
                    break

                loss = _imitation_loss(agent.policy, state_t,
                                       demo_action, legal_enc,
                                       agent.device, margin)
                if loss is not None:
                    batch_losses.append(loss)
                    n_samples += 1

                # Apply solver's move
                last_move = _apply_uci(board, demo_uci, last_move)

                # Apply opponent response (Moves[i+1]) if it exists
                if i + 1 < len(moves):
                    last_move = _apply_uci(board, moves[i + 1], last_move)

                # Batch update
                if len(batch_losses) >= batch_size:
                    _opt_step(opt, batch_losses, agent.policy)
                    batch_losses = []

            except Exception:
                n_skipped += 1
                puzzle_ok = False
                break

        if n_puzzles % log_every == 0:
            print(f"[stage 1] puzzles={n_puzzles}  samples={n_samples}  skipped={n_skipped}")
            # Periodic save mid-stage
            if save_path:
                agent.policy.eval(); agent.save(save_path.replace('.pth', f'_{n_puzzles}.pth')); agent.policy.train()

    # Flush remainder
    _opt_step(opt, batch_losses, agent.policy)
    agent.policy.eval()

    print(f"[stage 1] Puzzle pretraining complete: "
          f"puzzles={n_puzzles}, samples={n_samples}, skipped={n_skipped}")
    if save_path:
        agent.save(save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — RL episode (off-policy DQN vs Stockfish)
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(agent, stockfish, opp_color, max_plies=150,
                use_eval=True, learn_from_opponent=True):
    """
    One self-play game: agent plays agent.color, Stockfish plays opp_color.

    Each agent-turn transition is (s_t, a, r, s_{t+1}) where s_{t+1} is the
    position AFTER the opponent's reply — the MPC-MC single-agent MDP framing
    from Bertsekas et al. 2024.

    When learn_from_opponent=True, the opponent's transitions are also pushed
    into the shared replay buffer (with negated rewards), so the Q-network
    learns chess from both perspectives simultaneously.
    """
    board     = fresh_board()
    last_move = None
    s_t       = board_to_tensor(board, agent.color)
    total_r   = 0.0
    outcome   = 'draw'

    for _ in range(max_plies):
        # ── Agent's turn ─────────────────────────────────────────────────────
        legal = get_legal_moves(board, agent.color, last_move)
        if not legal:
            outcome = 'loss-no-moves' if is_in_check(board, agent.color, last_move) else 'draw'
            break

        mat_b = material_balance(board, agent.color)
        ev_b  = sf_eval(stockfish, board, agent.color, last_move) if use_eval else None

        fp, tp, act = agent.select_action(board, legal, training=True)
        piece  = board.grid[fp[0]][fp[1]]
        result = piece.move(fp, tp, board, last_move)
        if result == 'promotion': board.promote(tp[0], tp[1], Queen(agent.color))
        last_move = (board.grid[tp[0]][tp[1]], fp, tp)

        opp_check = is_in_check(board, opp_color, last_move)
        opp_mate  = opp_check and is_in_checkmate(board, opp_color, last_move)
        if opp_mate:
            agent.mem.push(s_t, act, agent.win_r, board_to_tensor(board, agent.color), True, [])
            total_r += agent.win_r; outcome = 'win'; break

        opp_stale = not opp_check and is_stalemate(board, opp_color, last_move)
        if opp_stale:
            agent.mem.push(s_t, act, agent.draw_r, board_to_tensor(board, agent.color), True, [])
            total_r += agent.draw_r; outcome = 'draw'; break

        # ── Opponent's turn ───────────────────────────────────────────────────
        s_opp = board_to_tensor(board, opp_color) if learn_from_opponent else None

        opp_mv = sf_move(stockfish, board, opp_color, last_move)
        if opp_mv is None:
            opp_legal = get_legal_moves(board, opp_color, last_move)
            if not opp_legal: break
            opp_mv = random.choice(opp_legal)

        ofrom, oto = opp_mv
        op    = board.grid[ofrom[0]][ofrom[1]]
        ores  = op.move(ofrom, oto, board, last_move)
        o_act = encode_action(ofrom, oto)
        if ores == 'promotion': board.promote(oto[0], oto[1], Queen(opp_color))
        last_move = (board.grid[oto[0]][oto[1]], ofrom, oto)

        self_check = is_in_check(board, agent.color, last_move)
        self_mate  = self_check and is_in_checkmate(board, agent.color, last_move)
        mat_a      = material_balance(board, agent.color)
        ev_a       = sf_eval(stockfish, board, agent.color, last_move) if use_eval else None
        step_r     = agent.step_reward(opp_check, self_check, mat_b, mat_a, ev_b, ev_a)
        next_t     = board_to_tensor(board, agent.color)

        # Opponent experience (negated reward)
        if learn_from_opponent and s_opp is not None:
            nla     = get_legal_moves(board, agent.color, last_move)
            nla_enc = [encode_action(fp2, tp2) for fp2, tp2 in nla]
            agent.mem.push(s_opp, o_act, -step_r,
                           board_to_tensor(board, opp_color), self_mate, nla_enc)

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
        nla_enc = [encode_action(fp2, tp2) for fp2, tp2 in nla]
        agent.mem.push(s_t, act, step_r, next_t, False, nla_enc)
        total_r += step_r
        s_t = next_t

    return outcome, total_r


# ─────────────────────────────────────────────────────────────────────────────
# TrainingManager — used by game.py Flask endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TrainingManager:
    def __init__(self):
        self.agent    = None
        self.sf_path  = r"C:\Users\eman2\Documents\GitHub\Project\Game\stockfish\stockfish-windows-x86-64-avx2.exe"
        self.sf_depth = 15
        self._thread  = None
        self.running  = False
        self._stop    = threading.Event()
        self._lock    = threading.Lock()
        self.wins = self.games = 0
        self._rh  = deque(maxlen=100)
        self.stats = {'episode': 0, 'win_rate': 0., 'avg_reward': 0., 'epsilon': 1., 'done': False}

    def configure(self, color='W', lr=3e-4, gamma=0.99, epsilon=1.0,
                  batch_size=256, sf_path=None, sf_depth=15, channels=256, num_res=10):
        self.agent = ChessQLearningAgent(color=color, lr=lr, gamma=gamma,
                                          epsilon=epsilon, batch_size=batch_size,
                                          channels=channels, num_res=num_res)
        if sf_path: self.sf_path = sf_path
        self.sf_depth = sf_depth
        self.wins = self.games = 0; self._rh.clear()
        self.stats = {'episode': 0, 'win_rate': 0., 'avg_reward': 0.,
                      'epsilon': self.agent.epsilon, 'done': False}

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
        sf = None
        if self.sf_path and _SF:
            try: sf = _SF(path=self.sf_path); sf.set_depth(self.sf_depth)
            except: sf = None
        opp = other_color(self.agent.color)
        for ep in range(1, episodes + 1):
            if self._stop.is_set(): break
            outcome, total_r = run_episode(self.agent, sf, opp,
                                            use_eval=(sf is not None),
                                            learn_from_opponent=True)
            for _ in range(8): self.agent.train_step()
            self.agent.decay_eps()
            if ep % 50 == 0:  self.agent.update_target()
            if ep % 200 == 0: self.agent.sched.step()
            with self._lock:
                self.games += 1
                if outcome == 'win': self.wins += 1
                self._rh.append(total_r)
                self.stats.update({'episode': ep, 'win_rate': self.wins / self.games,
                                   'avg_reward': sum(self._rh) / len(self._rh),
                                   'epsilon': self.agent.epsilon})
            if save_path and ep % save_interval == 0:
                self.agent.save(save_path)
        if save_path: self.agent.save(save_path)
        with self._lock: self.stats['done'] = True
        self.running = False

    def stop(self):
        self._stop.set()

    def get_stats(self):
        with self._lock: return dict(self.stats)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — staged training
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Staged chess RL training')
    parser.add_argument('--stage', choices=['openings', 'puzzles', 'rl', 'all'],
                        default='all', help='Which stage to run')
    parser.add_argument('--resume', action='store_true',
                        help='Load latest checkpoint before running chosen stage')
    parser.add_argument('--color', default='W', choices=['W', 'B'],
                        help='Color the agent plays during RL stage')
    parser.add_argument('--rl-episodes', type=int, default=200_000)
    parser.add_argument('--max-puzzles', type=int, default=150_000)
    parser.add_argument('--opening-epochs', type=int, default=5)
    args = parser.parse_args()

    os.makedirs('models', exist_ok=True)

    PATHS = {
        'openings': 'models/stage0_openings.pth',
        'puzzles':  'models/stage1_puzzles.pth',
        'rl':       'models/chess_bot.pth',
    }

    SF_PATH = r"C:\Users\eman2\Documents\GitHub\Project\Game\stockfish\stockfish-windows-x86-64-avx2.exe"

    # Create agent
    agent = ChessQLearningAgent(
        color=args.color,
        epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999,
        batch_size=256, channels=256, num_res=10,
    )

    # ── Stage 0: Opening pretraining ─────────────────────────────────────────
    if args.stage in ('openings', 'all'):
        if args.resume and os.path.exists(PATHS['openings']):
            agent.load(PATHS['openings'])
        pretrain_on_openings(
            agent,
            save_path=PATHS['openings'],
            epochs=args.opening_epochs,
            batch_size=128, lr=1e-3, margin=0.8,
            agent_color=None,   # train on both colors' opening moves
        )

    # ── Stage 1: Puzzle pretraining ───────────────────────────────────────────
    if args.stage in ('puzzles', 'all'):
        # Load openings checkpoint as starting point (if available)
        start_ck = PATHS['openings']
        if os.path.exists(start_ck):
            agent.load(start_ck)
        if args.resume and os.path.exists(PATHS['puzzles']):
            agent.load(PATHS['puzzles'])

        pretrain_on_puzzles(
            agent,
            save_path=PATHS['puzzles'],
            max_puzzles=args.max_puzzles,
            min_rating=900, max_rating=2200,
            themes_filter=None,   # None = use all themes
            batch_size=128, lr=5e-4, margin=0.8,
            log_every=5_000,
        )

    # ── Stage 2: RL training against Stockfish ────────────────────────────────
    if args.stage in ('rl', 'all'):
        # Load best available starting checkpoint
        for ck in (PATHS['puzzles'], PATHS['openings']):
            if os.path.exists(ck):
                agent.load(ck)
                break
        if args.resume and os.path.exists(PATHS['rl']):
            agent.load(PATHS['rl'])
        # After supervised pretraining, reset epsilon to a lower value so the
        # agent exploits what it learned while still exploring
        if not args.resume:
            agent.epsilon = 0.5

        sf = None
        if _SF:
            try: sf = _SF(path=SF_PATH); sf.set_depth(15)
            except Exception as e: print(f"[stage 2] Stockfish unavailable: {e}")

        opp_color = other_color(agent.color)
        ep = 0

        def _save_exit(*_):
            print('\n[stage 2] Interrupted — saving …')
            agent.save(PATHS['rl']); sys.exit(0)
        signal.signal(signal.SIGINT, _save_exit)

        print(f"[stage 2] Starting RL training: {args.rl_episodes} episodes, "
              f"agent={agent.color}, sf={'yes' if sf else 'random'}")

        while ep < args.rl_episodes:
            outcome, reward = run_episode(
                agent, sf, opp_color,
                use_eval=(sf is not None),
                learn_from_opponent=True,
            )
            for _ in range(8): agent.train_step()
            agent.decay_eps()
            if ep % 50  == 0: agent.update_target()
            if ep % 200 == 0: agent.sched.step()

            print(f"ep {ep:6d}: {outcome:<18s} r={reward:+7.2f}  ε={agent.epsilon:.4f}")

            if ep % 100 == 0:
                agent.save(PATHS['rl'])
                print("── saved ──")

            ep += 1

        agent.save(PATHS['rl'])
        print("[stage 2] RL training complete.")