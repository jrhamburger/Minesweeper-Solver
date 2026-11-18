"""Minesweeper as a constraint satisfaction problem - single-file version.

Sections of the plan:  1.1-1.3 solver | 1.4 simulator | 2 study design | 3 baselines B0-B3 | 4 evaluation

Usage
-----
  # debug / choose sampling settings on the dev split (50 boards per level)
  python minesweeper.py run --split dev --out results_dev --workers 4
  python minesweeper.py analyze results_dev

  # freeze settings, then run the test split once (300 boards/level x 4 configs = 3600 games)
  python minesweeper.py run --split test --out results_test --workers 4
  python minesweeper.py analyze results_test
  python minesweeper.py examples --level expert --n-examples 3 --out examples

  # sampling settings for 'run': --k 100 --max-draws 131072 --min-samples 10 --tie-eps 0.02
  #   restrict with --levels / --configs / --n-boards

Choices the plan left open
--------------------------
- Zero cells open their neighbours automatically, as in the real game.
- One "move" = one cell action; a move's time = time since the previous move.
- Fallback for non-frontier cells: global density (remaining mines / covered cells).
- Tie-break within 0.02 of the minimum: fewest neighbours (corner > edge > interior),
  then lowest estimate, then random.
- Random guessers (B0, B1) pick uniformly among covered, unflagged cells.
- Sampling stops at K kept layouts or after max_draws drawn layouts (default 131,072,
  about 0.5 s on a laptop) instead of after 0.5 s of wall-clock time. A clock limit
  makes the number of samples depend on machine load, so the same seed could give a
  different guess; a draw budget makes every game exactly reproducible.
- Dev seeds are (1, level, i), test seeds (2, level, i); the solver's own randomness
  uses a separate seed per configuration. Every configuration plays the same boards.

Requires numpy, pandas, matplotlib.
"""
import argparse
import csv
import math
import os
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


# ======================================================================
# GAME
# ======================================================================
# Standard difficulty levels: (rows, cols, mines)
LEVELS = {
    "beginner": (9, 9, 10),
    "intermediate": (16, 16, 40),
    "expert": (16, 30, 99),
}

COVERED = -1
FLAG = -2


def neighbor_table(rows, cols):
    """nbrs[i] = flat indices of the (up to 8) neighbours of flat cell i."""
    nbrs = []
    for r in range(rows):
        for c in range(cols):
            nbrs.append([
                (r + dr) * cols + (c + dc)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr or dc) and 0 <= r + dr < rows and 0 <= c + dc < cols
            ])
    return nbrs


class Game:
    def __init__(self, rows, cols, n_mines, seed):
        self.rows, self.cols, self.n_mines = rows, cols, n_mines
        self._rng = np.random.default_rng(seed)
        self._nbrs = neighbor_table(rows, cols)
        self._mines = None       # hidden; created on the first click
        self._counts = None
        self.revealed = np.zeros(rows * cols, dtype=bool)
        self.flagged = np.zeros(rows * cols, dtype=bool)
        self.lost = False
        self._safe_left = rows * cols - n_mines

    # ---------- state the agent may see ----------
    @property
    def won(self):
        return self._safe_left == 0 and not self.lost

    @property
    def over(self):
        return self.lost or self._safe_left == 0

    def mines_remaining(self):
        return self.n_mines - int(self.flagged.sum())

    def view(self):
        """Flat array: number 0-8 if revealed, FLAG if flagged, COVERED otherwise."""
        v = np.full(self.rows * self.cols, COVERED, dtype=np.int8)
        if self._counts is not None:
            v[self.revealed] = self._counts[self.revealed]
        v[self.flagged] = FLAG
        return v

    # ---------- actions ----------
    def reveal(self, i):
        """Reveal flat cell i. Returns the list of cells that changed.
        Zeros open their neighbours automatically, as in the real game."""
        if self.over:
            raise RuntimeError("game is over")
        if self.revealed[i] or self.flagged[i]:
            return []
        if self._mines is None:
            self._place_mines(i)
        if self._mines[i]:
            self.lost = True
            self.revealed[i] = True
            return [i]
        changed, stack = [], [i]
        while stack:
            j = stack.pop()
            if self.revealed[j] or self.flagged[j]:
                continue
            self.revealed[j] = True
            changed.append(j)
            if self._counts[j] == 0:
                stack.extend(k for k in self._nbrs[j]
                             if not self.revealed[k] and not self.flagged[k])
        self._safe_left -= len(changed)
        return changed

    def flag(self, i):
        if not self.revealed[i]:
            self.flagged[i] = True
        return [i]

    # ---------- setup ----------
    def _place_mines(self, first):
        forbidden = set(self._nbrs[first]) | {first}
        candidates = np.array([k for k in range(self.rows * self.cols)
                               if k not in forbidden])
        chosen = self._rng.choice(candidates, self.n_mines, replace=False)
        self._mines = np.zeros(self.rows * self.cols, dtype=bool)
        self._mines[chosen] = True
        self._counts = np.array([self._mines[nb].sum() for nb in self._nbrs],
                                dtype=np.int8)

    # ---------- ground truth: evaluation/logging ONLY, never given to the agent ----------
    def truth_is_mine(self, i):
        return bool(self._mines[i])

    def truth_mines(self):
        return self._mines.copy()


# ======================================================================
# SOLVER
# ======================================================================
@dataclass(frozen=True)
class SolverConfig:
    name: str
    use_stage1: bool
    use_stage2: bool
    guess: str                 # "random" or "mc"
    k_samples: int = 100       # K kept layouts
    max_draws: int = 131072    # layouts drawn per guess before giving up (32 batches;
                               # ~0.5 s on a laptop). A draw budget, not a clock, keeps
                               # every run identical for a given seed.
    min_samples: int = 10      # below this, use the local fallback
    tie_eps: float = 0.02      # tie window for the edge/corner heuristic
    batch: int = 4096          # layouts drawn per vectorised batch

    def with_settings(self, **kw):
        return SolverConfig(**{**self.__dict__, **kw})


CONFIGS = {
    "B0": SolverConfig("B0", use_stage1=False, use_stage2=False, guess="random"),
    "B1": SolverConfig("B1", use_stage1=True, use_stage2=True, guess="random"),
    "B2": SolverConfig("B2", use_stage1=True, use_stage2=False, guess="mc"),
    "B3": SolverConfig("B3", use_stage1=True, use_stage2=True, guess="mc"),
}


@dataclass
class Move:
    n: int            # move number within the game
    cell: int         # flat index (row = cell // cols, col = cell % cols)
    action: str       # "reveal" or "flag"
    stage: str        # "first", "stage1", "stage2", "guess"
    method: str       # for guesses: "random", "mc", "fallback"; else ""
    prob: float       # estimated mine probability (MC/fallback guesses only), else NaN
    n_samples: int    # kept layouts behind the estimate
    seconds: float    # time since the previous move (decision + action)


class Solver:
    def __init__(self, cfg: SolverConfig, seed, on_guess=None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.on_guess = on_guess   # optional callback(view, covered, probs, chosen, method)

    # ------------------------------------------------------------------ main loop
    def play(self, game):
        self.game = game
        self.cols = game.cols
        self.nbrs = neighbor_table(game.rows, game.cols)
        self.v = game.view()
        self.queue, self.in_queue = deque(), set()
        self.moves = []
        self._t = time.perf_counter()

        center = (game.rows // 2) * game.cols + game.cols // 2
        self._act(center, "reveal", "first")

        while not game.over:
            if self.cfg.use_stage1:
                self._stage1()                  # drains the queue
                if game.over:
                    break
            if self.cfg.use_stage2 and self._stage2():
                continue                        # new moves re-filled the queue
            self._guess()
        return self.moves

    def _act(self, cell, action, stage, method="", prob=float("nan"), n_samples=0):
        changed = (self.game.reveal(cell) if action == "reveal"
                   else self.game.flag(cell))
        now = time.perf_counter()
        self.moves.append(Move(len(self.moves), cell, action, stage, method,
                               prob, n_samples, now - self._t))
        self._t = now
        self.v = self.game.view()
        if self.cfg.use_stage1 and not self.game.over:
            # every changed constraint goes back on the queue
            for j in changed:
                for k in (j, *self.nbrs[j]):
                    if self.v[k] > 0 and k not in self.in_queue:
                        self.queue.append(k)
                        self.in_queue.add(k)

    # ------------------------------------------------------------------ constraints
    def _constraint(self, i):
        """For revealed number i: (covered neighbours, mines still unaccounted for)."""
        cov = [j for j in self.nbrs[i] if self.v[j] == COVERED]
        flags = sum(1 for j in self.nbrs[i] if self.v[j] == FLAG)
        return cov, int(self.v[i]) - flags

    def _all_constraints(self):
        out = []
        for i in np.flatnonzero(self.v >= 0):
            cov, rem = self._constraint(i)
            if cov:
                out.append((i, cov, rem))
        return out

    # ------------------------------------------------------------------ stage 1
    def _stage1(self):
        while self.queue and not self.game.over:
            i = self.queue.popleft()
            self.in_queue.discard(i)
            cov, rem = self._constraint(i)
            if not cov:
                continue
            if rem == len(cov):                 # every covered neighbour is a mine
                for j in cov:
                    if self.v[j] == COVERED:
                        self._act(j, "flag", "stage1")
            elif rem == 0:                      # all mines already flagged
                for j in cov:
                    if self.v[j] == COVERED and not self.game.over:
                        self._act(j, "reveal", "stage1")

    # ------------------------------------------------------------------ stage 2
    def _stage2(self):
        cons = {i: (frozenset(cov), rem) for i, cov, rem in self._all_constraints()}
        by_cell = {}
        for i, (cov, _) in cons.items():
            for j in cov:
                by_cell.setdefault(j, []).append(i)

        safe, mines, seen = set(), set(), set()
        for owners in by_cell.values():
            for a in owners:
                for b in owners:
                    if a == b or (a, b) in seen:
                        continue
                    seen.add((a, b))
                    (A, ra), (B, rb) = cons[a], cons[b]
                    if A < B:                   # A is a strict subset of B
                        diff, d = B - A, rb - ra
                        if d == 0:
                            safe |= diff
                        elif d == len(diff):
                            mines |= diff

        for j in sorted(mines):
            if self.v[j] == COVERED:
                self._act(j, "flag", "stage2")
        for j in sorted(safe):
            if self.v[j] == COVERED and not self.game.over:
                self._act(j, "reveal", "stage2")
        return bool(safe or mines)

    # ------------------------------------------------------------------ stage 3
    def _guess(self):
        covered = np.flatnonzero(self.v == COVERED)
        if self.cfg.guess == "random":
            j = int(self.rng.choice(covered))
            if self.on_guess:
                self.on_guess(self.v.copy(), covered, None, j, "random")
            self._act(j, "reveal", "guess", "random")
            return

        probs, method, n = self._estimate(covered)
        j = self._pick(covered, probs)
        if self.on_guess:
            self.on_guess(self.v.copy(), covered, probs, j, method)
        k = int(np.searchsorted(covered, j))
        self._act(j, "reveal", "guess", method, float(probs[k]), n)

    def _estimate(self, covered):
        """Rejection sampling: drop the remaining mines uniformly on covered cells,
        keep layouts consistent with every revealed number."""
        cfg, N = self.cfg, len(covered)
        m = self.game.mines_remaining()
        if m <= 0:
            return np.zeros(N), "mc", 0
        if m >= N:
            return np.ones(N), "mc", 0

        col = {cell: k for k, cell in enumerate(covered)}
        cons = self._all_constraints()
        A = np.zeros((len(cons), N), dtype=np.float32)
        b = np.zeros(len(cons), dtype=np.float32)
        for r, (_, cov, rem) in enumerate(cons):
            A[r, [col[j] for j in cov]] = 1
            b[r] = rem

        kept, n_kept, drawn = [], 0, 0
        rows = np.arange(cfg.batch)[:, None]
        while n_kept < cfg.k_samples and drawn < cfg.max_draws:
            drawn += cfg.batch
            keys = self.rng.random((cfg.batch, N))
            idx = np.argpartition(keys, m - 1, axis=1)[:, :m]
            X = np.zeros((cfg.batch, N), dtype=np.float32)
            X[rows, idx] = 1
            ok = np.all(X @ A.T == b, axis=1)
            if ok.any():
                kept.append(X[ok])
                n_kept += int(ok.sum())

        if n_kept >= cfg.min_samples:
            X = np.concatenate(kept)[:cfg.k_samples]
            return X.mean(axis=0), "mc", len(X)
        return self._fallback(covered, cons, col, m), "fallback", n_kept

    def _fallback(self, covered, cons, col, m):
        """Frontier cell: highest (unflagged mines / covered cells) among its numbers.
        Non-frontier cell: global density m / covered (our choice; the plan leaves it open)."""
        probs = np.full(len(covered), m / len(covered))
        frontier = np.zeros(len(covered), dtype=bool)
        local = np.zeros(len(covered))
        for _, cov, rem in cons:
            ratio = rem / len(cov)
            for j in cov:
                k = col[j]
                frontier[k] = True
                local[k] = max(local[k], ratio)
        probs[frontier] = local[frontier]
        return probs

    def _pick(self, covered, probs):
        """Lowest probability; within tie_eps of it, prefer corners, then edges."""
        cand = np.flatnonzero(probs <= probs.min() + self.cfg.tie_eps)
        n_nb = np.array([len(self.nbrs[covered[k]]) for k in cand])
        cand = cand[n_nb == n_nb.min()]
        cand = cand[probs[cand] == probs[cand].min()]
        return int(covered[self.rng.choice(cand)])


# ======================================================================
# EXPERIMENT
# ======================================================================
SPLIT_ID = {"dev": 1, "test": 2}          # dev and test seeds never overlap
LEVEL_ID = {"beginner": 0, "intermediate": 1, "expert": 2}
CONFIG_ID = {"B0": 0, "B1": 1, "B2": 2, "B3": 3}
DEFAULT_N = {"dev": 50, "test": 300}


def board_seed(split, level, idx):
    return [SPLIT_ID[split], LEVEL_ID[level], idx]


def play_one(job):
    split, level, idx, cfg = job
    seed = board_seed(split, level, idx)
    game = Game(*LEVELS[level], seed=seed)
    solver_seed = seed + [CONFIG_ID[cfg.name]]
    t0 = time.perf_counter()
    moves = Solver(cfg, solver_seed).play(game)
    seconds = time.perf_counter() - t0

    move_rows, wrong_certain = [], 0
    for mv in moves:
        is_mine = game.truth_is_mine(mv.cell)
        correct = is_mine if mv.action == "flag" else not is_mine
        certain = mv.stage in ("stage1", "stage2")
        wrong_certain += certain and not correct
        row = asdict(mv)
        row.update(config=cfg.name, level=level, board=idx,
                   row=mv.cell // game.cols, col=mv.cell % game.cols,
                   correct=int(correct), hit_mine=int(is_mine and mv.action == "reveal"))
        move_rows.append(row)

    guesses = [m for m in moves if m.stage == "guess"]
    game_row = dict(
        config=cfg.name, level=level, board=idx, seed="-".join(map(str, seed)),
        won=int(game.won), n_moves=len(moves),
        n_stage1=sum(m.stage == "stage1" for m in moves),
        n_stage2=sum(m.stage == "stage2" for m in moves),
        n_guesses=len(guesses),
        n_fallback=sum(m.method == "fallback" for m in guesses),
        wrong_certain=wrong_certain,
        safe_revealed=int(game.revealed.sum()) - int(game.lost),
        seconds=round(seconds, 5),
    )
    return game_row, move_rows


def experiment_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--n-boards", type=int, help="boards per level (default 50 dev / 300 test)")
    ap.add_argument("--levels", nargs="+", default=list(LEVELS))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--workers", type=int, default=1,
                    help=">1 runs games in parallel; per-game times may rise a little")
    ap.add_argument("--out", default=None)
    # sampling settings (choose these on dev, then freeze before the test run)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--max-draws", type=int, default=131072,
                    help="layouts drawn per guess before falling back (replaces the 0.5 s limit)")
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--tie-eps", type=float, default=0.02)
    args = ap.parse_args(argv)

    n = args.n_boards or DEFAULT_N[args.split]
    out = args.out or f"results_{args.split}"
    os.makedirs(out, exist_ok=True)
    overrides = dict(k_samples=args.k, max_draws=args.max_draws,
                     min_samples=args.min_samples, tie_eps=args.tie_eps)

    jobs = [(args.split, level, idx, CONFIGS[c].with_settings(**overrides))
            for c in args.configs for level in args.levels for idx in range(n)]
    print(f"{len(jobs)} games on the {args.split} split -> {out}/")

    games_f = open(os.path.join(out, "games.csv"), "w", newline="")
    moves_f = open(os.path.join(out, "moves.csv"), "w", newline="")
    gw = mw = None
    t0 = time.time()
    with Pool(args.workers) if args.workers > 1 else _Serial() as pool:
        for k, (g, ms) in enumerate(pool.imap(play_one, jobs, chunksize=1), 1):
            if gw is None:
                gw = csv.DictWriter(games_f, fieldnames=list(g)); gw.writeheader()
                mw = csv.DictWriter(moves_f, fieldnames=list(ms[0])); mw.writeheader()
            gw.writerow(g)
            mw.writerows(ms)
            if k % 25 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} games  ({time.time() - t0:.0f}s elapsed)", flush=True)
    games_f.close(); moves_f.close()

    with open(os.path.join(out, "settings.txt"), "w") as f:
        f.write(f"split={args.split} n_boards={n} levels={args.levels} configs={args.configs}\n")
        f.write(f"{overrides}\n")
    print("done")


class _Serial:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def imap(self, fn, it, chunksize=1): return map(fn, it)


# ======================================================================
# ANALYZE
# ======================================================================
LEVEL_ORDER = ["beginner", "intermediate", "expert"]
CONFIG_ORDER = ["B0", "B1", "B2", "B3"]
COLORS = {"B0": "#2a78d6", "B1": "#eb6834", "B2": "#1baf7a", "B3": "#eda100"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10,
})


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test on discordant pairs (paired boards)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def results_table(games):
    rows = []
    for (cfg, lvl), g in games.groupby(["config", "level"]):
        lo, hi = wilson(g.won.sum(), len(g))
        rows.append(dict(config=cfg, level=lvl, games=len(g), wins=int(g.won.sum()),
                         win_rate=g.won.mean(), ci_low=lo, ci_high=hi,
                         guesses_per_game=g.n_guesses.mean(),
                         fallback_share=(g.n_fallback.sum() / max(g.n_guesses.sum(), 1)),
                         sec_per_game=g.seconds.mean(),
                         wrong_certain=int(g.wrong_certain.sum())))
    t = pd.DataFrame(rows)
    t["level"] = pd.Categorical(t.level, LEVEL_ORDER, ordered=True)
    return t.sort_values(["level", "config"]).reset_index(drop=True)


def paired_tests(games, a="B3", b="B1"):
    out = []
    for lvl in LEVEL_ORDER:
        g = games[games.level == lvl].pivot(index="board", columns="config", values="won")
        if a not in g or b not in g:
            continue
        only_a = int(((g[a] == 1) & (g[b] == 0)).sum())
        only_b = int(((g[a] == 0) & (g[b] == 1)).sum())
        out.append(dict(level=lvl, **{f"{a}_only_wins": only_a, f"{b}_only_wins": only_b},
                        p_value=mcnemar_exact(only_a, only_b)))
    return pd.DataFrame(out)


def bar_chart(table, path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    levels = [l for l in LEVEL_ORDER if l in set(table.level)]
    cfgs = [c for c in CONFIG_ORDER if c in set(table.config)]
    w = 0.8 / len(cfgs)
    for k, cfg in enumerate(cfgs):
        sub = table[table.config == cfg].set_index("level").reindex(levels)
        x = np.arange(len(levels)) + (k - (len(cfgs) - 1) / 2) * w
        ax.bar(x, sub.win_rate, w - 0.02, color=COLORS[cfg], label=cfg,
               edgecolor=SURFACE, linewidth=1)
        ax.errorbar(x, sub.win_rate, yerr=[sub.win_rate - sub.ci_low, sub.ci_high - sub.win_rate],
                    fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
        for xi, wr, hi in zip(x, sub.win_rate, sub.ci_high):
            if wr == 0:
                ax.text(xi, hi + 0.01, "0%", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(np.arange(len(levels)), [l.capitalize() for l in levels])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Win rate (95% Wilson CI)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Win rate by difficulty and configuration", loc="left", fontsize=12, pad=24)
    ax.legend(frameon=False, ncol=len(cfgs), loc="lower left", bbox_to_anchor=(0, 1.0),
              borderaxespad=0, handlelength=1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def calibration(moves, path, bins=(0, .05, .1, .15, .2, .25, .3, .4, .5, .7, 1.0), min_count=10):
    g = moves[(moves.stage == "guess") & moves.prob.notna()].copy()
    g["bin"] = pd.cut(g.prob, bins, include_lowest=True)

    def curve(df):
        c = df.groupby("bin", observed=True).agg(est=("prob", "mean"), act=("hit_mine", "mean"),
                                                 n=("hit_mine", "size"))
        return c[c.n >= min_count]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    panels = [("By configuration", "config", {"B2": COLORS["B2"], "B3": COLORS["B3"]}),
              ("By estimate source (B2 + B3)", "method", {"mc": COLORS["B0"], "fallback": COLORS["B1"]})]
    for ax, (title, col, colors) in zip(axes, panels):
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="perfect calibration")
        for key, color in colors.items():
            c = curve(g[g[col] == key])
            if len(c):
                ax.plot(c.est, c.act, color=color, lw=2, marker="o", ms=8,
                        markeredgecolor=SURFACE, markeredgewidth=2,
                        label=f"{key} ({int(c.n.sum())} guesses)")
        top = max(0.3, float(g.prob.max()) if len(g) else 0.3)
        ax.set_xlim(0, min(1, top + 0.05)); ax.set_ylim(0, min(1, top + 0.05))
        ax.set_xlabel("Estimated mine probability")
        ax.set_title(title, loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("Actual mine rate")
    fig.suptitle(f"Calibration of guesses (bins with ≥{min_count} guesses)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return g


def analyze_main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    d = argv[0] if argv else "results_test"
    games = pd.read_csv(os.path.join(d, "games.csv"))
    moves = pd.read_csv(os.path.join(d, "moves.csv"))

    table = results_table(games)
    table.to_csv(os.path.join(d, "results_table.csv"), index=False)
    pd.set_option("display.width", 160)
    print("\n=== Results table ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    tests = paired_tests(games)
    if len(tests):
        print("\n=== B3 vs B1 on the same boards (exact McNemar) ===")
        print(tests.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        tests.to_csv(os.path.join(d, "b3_vs_b1.csv"), index=False)

    bar_chart(table, os.path.join(d, "win_rate_bar.png"))
    guesses = calibration(moves, os.path.join(d, "calibration.png"))

    # ---- success criteria (Section 4) ----
    print("\n=== Success criteria ===")
    n_wrong = int(games.wrong_certain.sum())
    print(f"1. Wrong certain moves across {len(games)} games: {n_wrong}  -> {'PASS' if n_wrong == 0 else 'FAIL'}")
    wr = table.set_index(["config", "level"]).win_rate
    for lvl in ("intermediate", "expert"):
        if ("B3", lvl) in wr and ("B1", lvl) in wr:
            ok = wr["B3", lvl] > wr["B1", lvl]
            print(f"2. {lvl}: B3 {wr['B3', lvl]:.3f} vs B1 {wr['B1', lvl]:.3f}  -> {'PASS' if ok else 'FAIL'}")
    sec = table[table.config == "B3"].set_index("level").sec_per_game
    for lvl in ("beginner", "intermediate"):
        if lvl in sec:
            print(f"3. B3 {lvl}: {sec[lvl]:.3f} s/game  -> {'PASS' if sec[lvl] < 1 else 'FAIL'}")
    if len(guesses) > 2:
        r = np.corrcoef(guesses.prob, guesses.hit_mine)[0, 1]
        lo = guesses[guesses.prob <= guesses.prob.median()].hit_mine.mean()
        hi = guesses[guesses.prob > guesses.prob.median()].hit_mine.mean()
        print(f"4. Guesses: corr(estimate, hit) = {r:.3f}; mine rate below/above median estimate "
              f"= {lo:.3f} / {hi:.3f}  -> {'PASS' if hi > lo and r > 0 else 'FAIL'}")
    print(f"\nWrote results_table.csv, b3_vs_b1.csv, win_rate_bar.png, calibration.png to {d}/")


# ======================================================================
# EXAMPLES
# ======================================================================
# one-hue sequential ramp: light (safe) -> dark (likely mine)
CMAP = LinearSegmentedColormap.from_list("risk", ["#f3eee3", "#eda100", "#8a4b00"])


def draw(view, rows, cols, covered, probs, chosen, method, truth, title, path):
    fig, ax = plt.subplots(figsize=(cols * 0.38 + 1.6, rows * 0.38 + 0.9))
    grid = np.full((rows, cols), np.nan)
    if probs is not None:
        grid.flat[covered] = probs
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=1)
    for i, v in enumerate(view):
        r, c = divmod(i, cols)
        if v >= 0:
            ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="#ffffff", zorder=1))
            if v > 0:
                ax.text(c, r, str(v), ha="center", va="center", fontsize=8, color="#0b0b0b", zorder=2)
        elif v == FLAG:
            ax.text(c, r, "F", ha="center", va="center", fontsize=8, color="#e34948",
                    weight="bold", zorder=2)
    r, c = divmod(chosen, cols)
    ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, ec="#2a78d6", lw=2.5, zorder=3))
    ax.set_xticks(np.arange(-.5, cols), minor=True); ax.set_yticks(np.arange(-.5, rows), minor=True)
    ax.grid(which="minor", color="#d0cfca", lw=0.5); ax.tick_params(which="both", length=0)
    ax.set_xticks([]); ax.set_yticks([])
    outcome = "MINE (lost)" if truth[chosen] else "safe"
    p = "" if probs is None else f", est. p = {probs[np.searchsorted(covered, chosen)]:.3f}"
    ax.set_title(f"{title}\nguess at ({r}, {c}) via {method}{p} -> {outcome}", fontsize=9, loc="left")
    if probs is not None:
        fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1)), ax=ax,
                     fraction=0.03, pad=0.02, label="estimated mine probability")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def examples_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--level", default="intermediate", choices=list(LEVELS))
    ap.add_argument("--n-examples", type=int, default=3,
                    help="number of boards (that needed at least one guess) to illustrate")
    ap.add_argument("--max-per-board", type=int, default=2)
    ap.add_argument("--out", default="examples")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    rows, cols, _ = LEVELS[args.level]
    shown = 0
    for b in range(1000):
        if shown >= args.n_examples:
            break
        seed = board_seed(args.split, args.level, b)
        game = Game(rows, cols, LEVELS[args.level][2], seed=seed)
        captured = []

        def on_guess(view, covered, probs, chosen, method):
            if len(captured) < args.max_per_board:
                captured.append((view, covered, probs, chosen, method))

        Solver(CONFIGS["B3"], seed + [CONFIG_ID["B3"]], on_guess=on_guess).play(game)
        truth = game.truth_mines()
        result = "won" if game.won else "lost"
        for k, (view, covered, probs, chosen, method) in enumerate(captured):
            path = os.path.join(args.out, f"{args.level}_board{b}_guess{k + 1}.png")
            title = f"{args.level.capitalize()} board {b}, guess {k + 1} ({result} game)"
            draw(view, rows, cols, covered, probs, chosen, method, truth, title, path)
            print("wrote", path)
        shown += bool(captured)


# ======================================================================
# COMMAND LINE
# ======================================================================
COMMANDS = {"run": experiment_main, "analyze": analyze_main, "examples": examples_main}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(f"usage: python {os.path.basename(sys.argv[0])} {{run,analyze,examples}} [options]")
    COMMANDS[sys.argv[1]](sys.argv[2:])