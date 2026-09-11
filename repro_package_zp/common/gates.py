"""common/gates.py — structural gate checkers (verbatim extraction).

Provenance (M2 extraction 2026-09-02): repro/r92_gate_sampler.py lines
47-179 (build_train / classify / sample_pi / verify), split into pure
functions.  The within-sum + train-closed + swap-equivariant family
serves K09 (R92 fixed/resamp) and K10 (R84/R86 persistence).

Check semantics (r92_gate_sampler.py docstring, p=113 odd):
  V1  bijection on T (pi(T)=T);
  V2  within-sum: y[pi[i]] == y[i] for all i in T;
  V3  fixed points exactly = registered unavoidable set U
      (diag_T ∪ half-singletons), no others;
  V4  swap-equivariance on T (rows with partner in T: pi[S(i)]==S[pi[i]]);
  V6  closure extension to C = T ∪ S(T) is a bijection on C and
      swap-equivariant on all of C (audit-only, never forwarded).

Hard-won lessons encoded here:
  - keyed derangement: perm = rng.permutation(k); pi[d[perm]] =
    d[roll(perm, 1)] -- key = the permuted order itself; a sort-keyed
    assignment creates fixed points (R87 gate lesson);
  - batch<->pair correspondence consumes id_train UNSORTED (§6cz-prime).
"""
import numpy as np

from . import task


class ZPIndexing:
    """Precomputed index arrays for the zp table (r92_gate_sampler.py
    lines 47-58, verbatim logic; Y_ARR via task tables)."""

    def __init__(self):
        task.set_group('zp', 113)
        P = task.GROUP['p']
        n_pairs = task.N_PAIRS
        self.P = P
        self.N_PAIRS = n_pairs
        IDX = np.arange(n_pairs, dtype=np.int64)
        A_ARR = IDX % P
        B_ARR = IDX // P
        Y_ARR = (A_ARR + B_ARR) % P
        S_ARR = (IDX % P) * P + IDX // P          # swap: (a,b) -> (b,a)
        DIAG = IDX[A_ARR == B_ARR]
        OFF = IDX[A_ARR != B_ARR]
        ORB_REP = np.minimum(OFF, S_ARR[OFF])
        self.IDX, self.A_ARR, self.B_ARR, self.Y_ARR = IDX, A_ARR, B_ARR, Y_ARR
        self.S_ARR, self.DIAG, self.OFF = S_ARR, DIAG, OFF
        self.ORB_REPS = np.unique(ORB_REP)


def build_train(zp, seed, split_frac=0.30):
    """r92_gate_sampler.py lines 67-74 (verbatim; equals task.make_split
    construction, kept separate to pin the harness reference)."""
    n_train = int(round(zp.N_PAIRS * split_frac))
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    perm = rng.permutation(zp.N_PAIRS)
    id_train = perm[:n_train]                 # unsorted (harness ref)
    tr = np.sort(id_train)
    tr_mask = np.zeros(zp.N_PAIRS, dtype=bool)
    tr_mask[tr] = True
    return id_train, tr, tr_mask


def classify(zp, tr_mask):
    """r92_gate_sampler.py lines 77-101 (verbatim logic)."""
    P = zp.P
    full = {c: [] for c in range(P)}
    half = {c: [] for c in range(P)}
    diag_T = zp.DIAG[tr_mask[zp.DIAG]]
    for rep in zp.ORB_REPS:
        i, j = int(rep), int(zp.S_ARR[rep])
        mi, mj = bool(tr_mask[i]), bool(tr_mask[j])
        c = int(zp.Y_ARR[rep])
        if mi and mj:
            full[c].append(rep)
        elif mi:
            half[c].append(i)
        elif mj:
            half[c].append(j)
    U = set(int(x) for x in diag_T)
    half_single = []
    for c in range(P):
        if len(half[c]) == 1:
            U.add(half[c][0])
            half_single.append(half[c][0])
    return full, half, set(int(x) for x in diag_T), half_single


def sample_pi(zp, seed, draw, full, half):
    """r92_gate_sampler.py lines 104-139 (verbatim).
    fixed arm: (seed, draw=0) from stream 7000+seed;
    resampled: stream 7500+seed, draw=t."""
    if draw == 0:
        rng = np.random.default_rng(7000 + seed)
    else:
        rng = np.random.default_rng(np.random.SeedSequence(
            [7500 + seed, draw]))
    IDX, S_ARR = zp.IDX, zp.S_ARR
    pi = IDX.copy()          # identity outside T (unused)
    n_orbit_flip = 0
    for c in range(zp.P):
        fo = full[c]
        if len(fo) >= 2:
            k = len(fo)
            permk = rng.permutation(k)
            tgt = np.roll(permk, 1)
            orient = rng.integers(0, 2, size=k)
            for m in range(k):
                i = int(fo[permk[m]])
                j = int(fo[tgt[m]])
                if orient[m] == 0:
                    pi[i], pi[S_ARR[i]] = j, S_ARR[j]
                else:
                    pi[i], pi[S_ARR[i]] = S_ARR[j], j
        elif len(fo) == 1:
            i = int(fo[0])
            pi[i], pi[S_ARR[i]] = int(S_ARR[i]), i   # forced self-flip
            n_orbit_flip += 1
        hh = half[c]
        if len(hh) >= 2:
            k = len(hh)
            permh = rng.permutation(k)
            pi[np.asarray(hh)[permh]] = np.asarray(hh)[
                np.roll(permh, 1)]
        # len==1: stays identity (unavoidable fixed, in U)
    return pi, n_orbit_flip


def verify_assignment(zp, pi, tr, tr_mask, U):
    """r92_gate_sampler.py verify() lines 142-179 (verbatim checks,
    returns dict of booleans; U must be precomputed via classify)."""
    S_ARR = zp.S_ARR
    Y_ARR = zp.Y_ARR
    N_PAIRS = zp.N_PAIRS
    out = {}
    # V1 bijection on T
    out['V1_bij'] = sorted(pi[tr].tolist()) == sorted(tr.tolist())
    # V2 within-sum
    out['V2_wsum'] = bool(np.all(Y_ARR[pi[tr]] == Y_ARR[tr]))
    # V3 fixed points exactly U
    fp = set(int(x) for x in tr[pi[tr] == tr])
    out['V3_fp_eq_U'] = (fp == U)
    # V4 equivariance on T (rows with partner in T)
    in_T_of = np.zeros(N_PAIRS, dtype=bool)
    in_T_of[tr] = True
    m = in_T_of[S_ARR[tr]]
    lhs = pi[S_ARR[tr[m]]]
    rhs = S_ARR[pi[tr[m]]]
    out['V4_eq'] = bool(np.all(lhs == rhs))
    # V6 closure extension audit
    ext = {int(tr[k]): int(pi[tr[k]]) for k in range(len(tr))}
    for k in range(len(tr)):
        i = int(tr[k])
        if not in_T_of[S_ARR[i]]:
            ext[S_ARR[i]] = int(S_ARR[pi[i]])
    C = set(ext.keys())
    v6a = set(ext.values()) == C
    v6b = True
    for i in C:
        Si = int(S_ARR[i])
        if Si in ext and ext[Si] != int(S_ARR[ext[i]]):
            v6b = False
            break
    out['V6_closure'] = bool(v6a and v6b)
    out['PASS'] = all(out.values())
    return out
