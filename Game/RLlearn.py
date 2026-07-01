import os, copy, random, threading, sys
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

PIECE_TYPES  = ['Pawn','Knight','Bishop','Rook','Queen','King']
PIECE_VALUES = {'Pawn':1,'Knight':3,'Bishop':3,'Rook':5,'Queen':9,'King':0}
NUM_ACTIONS  = 64 * 64


def other_color(c): return 'B' if c == 'W' else 'W'


def board_to_tensor(board, side_to_move):
    planes = torch.zeros(13, 8, 8, dtype=torch.float32)
    for r in range(8):
        for c in range(8):
            p = board.grid[r][c]
            if p == ' ': continue
            idx = PIECE_TYPES.index(p.name) + (0 if p.color == 'W' else 6)
            planes[idx, r, c] = 1.0
    planes[12] = 1.0 if side_to_move == 'W' else 0.0
    return planes


def encode_action(fp, tp): return (fp[0]*8+fp[1])*64 + (tp[0]*8+tp[1])


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


# ── Candidate squares per piece type (avoids brute-forcing all 64) ────────

def _pawn_cands(r, c, color):
    d = 1 if color=='W' else -1; sr = 1 if color=='W' else 6
    cands = [(r+d,c),(r+d,c-1),(r+d,c+1)]
    if r == sr: cands.append((r+2*d,c))
    return [(rr,cc) for rr,cc in cands if InBounds(rr,cc)]

def _knight_cands(r, c):
    return [(r+dr,c+dc) for dr,dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)] if InBounds(r+dr,c+dc)]

def _slide(r, c, dirs):
    out=[]
    for dr,dc in dirs:
        for i in range(1,8):
            rr,cc=r+dr*i,c+dc*i
            if not InBounds(rr,cc): break
            out.append((rr,cc))
    return out

def _bishop_cands(r,c): return _slide(r,c,[(-1,-1),(-1,1),(1,-1),(1,1)])
def _rook_cands(r,c):   return _slide(r,c,[(-1,0),(1,0),(0,-1),(0,1)])
def _queen_cands(r,c):  return _bishop_cands(r,c)+_rook_cands(r,c)
def _king_cands(r,c):
    out=[(r+dr,c+dc) for dr in(-1,0,1) for dc in(-1,0,1) if (dr or dc) and InBounds(r+dr,c+dc)]
    out+=[(r,c+2),(r,c-2)]
    return [p for p in out if InBounds(*p)]

_CANDS = {
    'Pawn':   _pawn_cands,
    'Knight': lambda r,c,col: _knight_cands(r,c),
    'Bishop': lambda r,c,col: _bishop_cands(r,c),
    'Rook':   lambda r,c,col: _rook_cands(r,c),
    'Queen':  lambda r,c,col: _queen_cands(r,c),
    'King':   lambda r,c,col: _king_cands(r,c),
}


def get_legal_moves(board, color, last_move=None):
    legal = []
    for r in range(8):
        for c in range(8):
            p = board.grid[r][c]
            if p == ' ' or p.color != color: continue
            for (tr,tc) in _CANDS[p.name](r, c, color):
                try:
                    tb = copy.deepcopy(board)
                    tb.grid[r][c].move((r,c),(tr,tc),tb,last_move)
                except: continue
                if not is_in_check(tb, color, last_move):
                    legal.append(((r,c),(tr,tc)))
    return legal


def fresh_board():
    b = Board()
    b.grid[0][0]=Rook("W"); b.grid[0][1]=Knight("W"); b.grid[0][2]=Bishop("W")
    b.grid[0][3]=Queen("W"); b.grid[0][4]=King("W");  b.grid[0][5]=Bishop("W")
    b.grid[0][6]=Knight("W"); b.grid[0][7]=Rook("W")
    for i in range(8): b.grid[1][i]=Pawn("W")
    b.grid[7][0]=Rook("B"); b.grid[7][1]=Knight("B"); b.grid[7][2]=Bishop("B")
    b.grid[7][3]=Queen("B"); b.grid[7][4]=King("B");  b.grid[7][5]=Bishop("B")
    b.grid[7][6]=Knight("B"); b.grid[7][7]=Rook("B")
    for i in range(8): b.grid[6][i]=Pawn("B")
    return b


# ── Stockfish helpers ──────────────────────────────────────────────────────

def local_board_to_fen(board, side_to_move, last_move=None):
    rows = []
    for r in range(7, -1, -1):        # rank 8 down to rank 1
        s, empty = "", 0
        for c in range(8):
            p = board.grid[r][c]
            if p == ' ': empty += 1
            else:
                if empty: s += str(empty); empty = 0
                s += p.fen
        if empty: s += str(empty)
        rows.append(s)
    placement = "/".join(rows)
    stm = 'w' if side_to_move == 'W' else 'b'

    castling = ""
    wk = board.grid[0][4]
    if wk != ' ' and wk.name=='King' and wk.color=='W' and not wk.history:
        wr,wl = board.grid[0][7], board.grid[0][0]
        if wr != ' ' and wr.name=='Rook' and wr.color=='W' and not wr.history: castling+="K"
        if wl != ' ' and wl.name=='Rook' and wl.color=='W' and not wl.history: castling+="Q"
    bk = board.grid[7][4]
    if bk != ' ' and bk.name=='King' and bk.color=='B' and not bk.history:
        br,bl = board.grid[7][7], board.grid[7][0]
        if br != ' ' and br.name=='Rook' and br.color=='B' and not br.history: castling+="k"
        if bl != ' ' and bl.name=='Rook' and bl.color=='B' and not bl.history: castling+="q"
    if not castling: castling = "-"

    ep = "-"
    if last_move:
        lm_piece,(fr,fc),(tr,tc) = last_move
        if lm_piece.name=='Pawn' and abs(fr-tr)==2:
            ep = f"{chr(ord('a')+fc)}{(fr+tr)//2+1}"

    return f"{placement} {stm} {castling} {ep} 0 1"


def uci_to_pos(uci):
    return (int(uci[1])-1, ord(uci[0])-ord('a')), (int(uci[3])-1, ord(uci[2])-ord('a'))


def sf_move(sf, board, color, last_move=None):
    if sf is None: return None
    try:
        sf.set_fen_position(local_board_to_fen(board, color, last_move), do_validation=False)
        uci = sf.get_best_move()
        return uci_to_pos(uci) if uci else None
    except: return None


def sf_eval(sf, board, color, last_move=None):
    if sf is None: return None
    try:
        sf.set_fen_position(local_board_to_fen(board, color, last_move), do_validation=False)
        r = sf.get_evaluation()
        return float(r['value']) if r['type']=='cp' else (10000.0 if r['value']>0 else -10000.0)
    except: return None


# ── Model ─────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.net(x) + x)


class QNetwork(nn.Module):
    """Deep residual Q-network (AlphaZero-style trunk, Q-learning head)."""
    def __init__(self, num_actions=NUM_ACTIONS, channels=256, num_res=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(13, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(*[ResBlock(channels) for _ in range(num_res)])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels*8*8, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512),          nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, num_actions),
        )
    def forward(self, x):
        return self.head(self.res(self.stem(x)))


class ReplayBuffer:
    def __init__(self, cap=40000):
        self.buf = deque(maxlen=cap)
    def push(self, *args): self.buf.append(args)
    def sample(self, n):   return random.sample(self.buf, n)
    def __len__(self):     return len(self.buf)


# ── Agent ─────────────────────────────────────────────────────────────────

class ChessQLearningAgent:
    def __init__(self, color='W', lr=3e-4, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.997, batch_size=256,
                 buf_size=40000, win_r=20.0, draw_r=2.0, loss_r=-20.0,
                 check_b=0.3, check_p=0.3, mat_w=1.0, eval_w=0.2,
                 channels=256, num_res=10, device=None):
        self.color = color
        self.gamma, self.epsilon = gamma, epsilon
        self.epsilon_min, self.epsilon_decay = epsilon_min, epsilon_decay
        self.batch_size = batch_size
        self.win_r, self.draw_r, self.loss_r = win_r, draw_r, loss_r
        self.check_b, self.check_p = check_b, check_p
        self.mat_w, self.eval_w = mat_w, eval_w

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        self.policy = QNetwork(channels=channels, num_res=num_res).to(self.device)
        self.target = QNetwork(channels=channels, num_res=num_res).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.opt = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=1000, eta_min=1e-5)
        self.mem = ReplayBuffer(buf_size)

    def select_action(self, board, legal, training=True):
        if not legal: return None
        if training and random.random() < self.epsilon:
            fp, tp = random.choice(legal)
        else:
            s = board_to_tensor(board, self.color).unsqueeze(0).to(self.device)
            with torch.no_grad(): qs = self.policy(s).squeeze(0)
            fp, tp = max(legal, key=lambda m: qs[encode_action(*m)].item())
        return fp, tp, encode_action(fp, tp)

    def step_reward(self, opp_check, self_check, mat_b, mat_a, ev_b, ev_a):
        r = self.check_b * opp_check - self.check_p * self_check
        r += self.mat_w * (mat_a - mat_b)
        if ev_b is not None and ev_a is not None:
            r += self.eval_w * max(-2.0, min(2.0, (ev_a - ev_b)/100.0))
        return r

    def train_step(self):
        if len(self.mem) < self.batch_size: return None
        batch = self.mem.sample(self.batch_size)
        S,A,R,S2,D,LA = zip(*batch)
        S  = torch.stack(S).to(self.device)
        A  = torch.tensor(A, dtype=torch.long, device=self.device)
        R  = torch.tensor(R, dtype=torch.float32, device=self.device)
        S2 = torch.stack(S2).to(self.device)
        D  = torch.tensor(D, dtype=torch.float32, device=self.device)
        q  = self.policy(S).gather(1, A.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq = self.target(S2)
            mx = torch.zeros(len(batch), device=self.device)
            for i,la in enumerate(LA):
                if la and D[i].item()==0:
                    idx = torch.tensor(la, dtype=torch.long, device=self.device)
                    mx[i] = nq[i, idx].max()
            tgt = R + self.gamma * mx * (1-D)
        loss = nn.functional.huber_loss(q, tgt)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()
        return loss.item()

    def update_target(self): self.target.load_state_dict(self.policy.state_dict())
    def decay_eps(self):     self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        torch.save({'policy': self.policy.state_dict(), 'target': self.target.state_dict(),
                    'epsilon': self.epsilon, 'color': self.color}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck['policy'])
        self.target.load_state_dict(ck['target'])
        self.epsilon = ck.get('epsilon', self.epsilon)
        self.color   = ck.get('color',   self.color)


# ── Episode ───────────────────────────────────────────────────────────────

def run_episode(agent, stockfish, opp_color, max_plies=150, use_eval=True,
                learn_from_opponent=True):
    """
    Agent plays agent.color; Stockfish plays opp_color (nominal opponent per
    the Bertsekas MPC-MC framing).  When learn_from_opponent=True, the
    opponent's transitions are also pushed into the replay buffer so the
    network learns from both sides.
    """
    board     = fresh_board()
    last_move = None
    s_tensor  = board_to_tensor(board, agent.color)
    total_r   = 0.0
    outcome   = 'draw'

    for _ in range(max_plies):
        # ── Agent's turn ─────────────────────────────────────────────────
        legal = get_legal_moves(board, agent.color, last_move)
        if not legal:
            outcome = 'loss-no-moves' if is_in_check(board, agent.color, last_move) else 'draw'
            break

        mat_b  = material_balance(board, agent.color)
        ev_b   = sf_eval(stockfish, board, agent.color, last_move) if use_eval else None

        fp, tp, act = agent.select_action(board, legal, training=True)
        piece  = board.grid[fp[0]][fp[1]]
        res    = piece.move(fp, tp, board, last_move)
        if res == 'promotion': board.promote(tp[0], tp[1], Queen(agent.color))
        last_move = (board.grid[tp[0]][tp[1]], fp, tp)

        opp_check = is_in_check(board, opp_color, last_move)
        opp_mate  = opp_check and is_in_checkmate(board, opp_color, last_move)
        if opp_mate:
            r = agent.win_r
            agent.mem.push(s_tensor, act, r, board_to_tensor(board, agent.color), True, [])
            total_r += r; outcome = 'win'; break

        opp_stale = (not opp_check) and is_stalemate(board, opp_color, last_move)
        if opp_stale:
            r = agent.draw_r
            agent.mem.push(s_tensor, act, r, board_to_tensor(board, agent.color), True, [])
            total_r += r; outcome = 'draw'; break

        # ── Opponent's turn ───────────────────────────────────────────────
        s_opp_before = board_to_tensor(board, opp_color) if learn_from_opponent else None

        opp_mv = sf_move(stockfish, board, opp_color, last_move)
        if opp_mv is None:
            opp_legal = get_legal_moves(board, opp_color, last_move)
            if not opp_legal: break
            opp_mv = random.choice(opp_legal)

        ofrom, oto = opp_mv
        op      = board.grid[ofrom[0]][ofrom[1]]
        ores    = op.move(ofrom, oto, board, last_move)
        opp_act = encode_action(ofrom, oto)
        if ores == 'promotion': board.promote(oto[0], oto[1], Queen(opp_color))
        last_move = (board.grid[oto[0]][oto[1]], ofrom, oto)

        self_check  = is_in_check(board, agent.color, last_move)
        self_mate   = self_check and is_in_checkmate(board, agent.color, last_move)
        mat_a       = material_balance(board, agent.color)
        ev_a        = sf_eval(stockfish, board, agent.color, last_move) if use_eval else None
        step_r      = agent.step_reward(opp_check, self_check, mat_b, mat_a, ev_b, ev_a)
        next_tensor = board_to_tensor(board, agent.color)

        # ── Store opponent experience (learn_from_opponent) ───────────────
        if learn_from_opponent and s_opp_before is not None:
            # Opponent's reward is the negative of the agent's step reward
            opp_reward  = -step_r
            next_legal_a = get_legal_moves(board, agent.color, last_move)
            nla_enc = [encode_action(f,t) for f,t in next_legal_a]
            agent.mem.push(s_opp_before, opp_act, opp_reward,
                           board_to_tensor(board, opp_color),
                           self_mate, nla_enc)

        if self_mate:
            step_r += agent.loss_r
            agent.mem.push(s_tensor, act, step_r, next_tensor, True, [])
            total_r += step_r; outcome = 'loss'; break

        self_stale = (not self_check) and is_stalemate(board, agent.color, last_move)
        if self_stale:
            step_r += agent.draw_r
            agent.mem.push(s_tensor, act, step_r, next_tensor, True, [])
            total_r += step_r; outcome = 'draw'; break

        nla     = get_legal_moves(board, agent.color, last_move)
        nla_enc = [encode_action(f,t) for f,t in nla]
        agent.mem.push(s_tensor, act, step_r, next_tensor, False, nla_enc)
        total_r  += step_r
        s_tensor  = next_tensor

    return outcome, total_r


# ── Training manager (called by game.py Flask endpoints) ──────────────────

class TrainingManager:
    def __init__(self):
        self.agent = None
        self.sf_path  = r"C:\Users\eman2\Documents\GitHub\Project\Game\stockfish\stockfish-windows-x86-64-avx2.exe"
        self.sf_depth = 15
        self._thread  = None
        self.running  = False
        self._stop    = threading.Event()
        self._lock    = threading.Lock()
        self.wins = self.games = 0
        self._rh  = deque(maxlen=100)
        self.stats = {'episode':0,'win_rate':0.,'avg_reward':0.,'epsilon':1.,'done':False}

    def configure(self, color='W', lr=3e-4, gamma=0.99, epsilon=1.0,
                  batch_size=256, sf_path=None, sf_depth=15,
                  channels=256, num_res=10):
        self.agent = ChessQLearningAgent(color=color, lr=lr, gamma=gamma,
                                          epsilon=epsilon, batch_size=batch_size,
                                          channels=channels, num_res=num_res)
        if sf_path: self.sf_path = sf_path
        self.sf_depth = sf_depth
        self.wins = self.games = 0; self._rh.clear()
        self.stats = {'episode':0,'win_rate':0.,'avg_reward':0.,
                      'epsilon':self.agent.epsilon,'done':False}

    def load_agent(self, path):
        if self.agent is None: self.configure()
        self.agent.load(path)

    def save_agent(self, path):
        if self.agent is None: raise RuntimeError("No agent")
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
        if self.sf_path and Stockfish:
            try: sf = Stockfish(path=self.sf_path); sf.set_depth(self.sf_depth)
            except: sf = None
        opp = other_color(self.agent.color)
        for ep in range(1, episodes+1):
            if self._stop.is_set(): break
            outcome, total_r = run_episode(self.agent, sf, opp,
                                            use_eval=(sf is not None),
                                            learn_from_opponent=True)
            for _ in range(8): self.agent.train_step()
            self.agent.decay_eps()
            if ep % 50 == 0: self.agent.update_target()
            if ep % 200 == 0: self.agent.sched.step()
            with self._lock:
                self.games += 1
                if outcome == 'win': self.wins += 1
                self._rh.append(total_r)
                self.stats.update({'episode':ep,'win_rate':self.wins/self.games,
                                   'avg_reward':sum(self._rh)/len(self._rh),
                                   'epsilon':self.agent.epsilon})
                print(self.stats)
            if save_path and ep % save_interval == 0:
                self.agent.save(save_path)
        if save_path: self.agent.save(save_path)
        with self._lock: self.stats['done'] = True
        self.running = False

    def stop(self):     self._stop.set()
    def get_stats(self):
        with self._lock: return dict(self.stats)


# ── Standalone training entry point ───────────────────────────────────────

if __name__ == '__main__':
    import signal
    load_path = save_path = "models/chess_bot.pth"
    os.makedirs("models", exist_ok=True)

    agent = ChessQLearningAgent(color='W', epsilon=1.0, epsilon_min=0.05,
                                 epsilon_decay=0.997, batch_size=256,
                                 channels=256, num_res=10)
    if os.path.exists(load_path):
        print(f"Resuming from {load_path}")
        agent.load(load_path)
    else:
        print("Starting fresh")

    sf = Stockfish(path=r"C:\Users\eman2\Documents\GitHub\Project\Game\stockfish\stockfish-windows-x86-64-avx2.exe")
    sf.set_depth(15)

    ep = 0
    def _save_exit(*_):
        print(f"\nSaving to {save_path}..."); agent.save(save_path); sys.exit(0)
    signal.signal(signal.SIGINT, _save_exit)

    while ep < 50000:
        outcome, reward = run_episode(agent, sf, 'B', use_eval=True, learn_from_opponent=True)
        for _ in range(8): agent.train_step()
        agent.decay_eps()
        if ep % 50  == 0: agent.update_target()
        if ep % 200 == 0: agent.sched.step()
        print(f"ep {ep}: {outcome}  r={reward:.2f}  eps={agent.epsilon:.3f}")
        if ep % 20 == 0: agent.save(save_path); print("── saved ──")
        ep += 1