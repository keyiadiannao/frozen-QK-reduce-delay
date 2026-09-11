"""common/task.py — Zp abeq task construction (verbatim extraction).

Provenance (M2 extraction 2026-09-02):
  - constants N_ROT/N_ELEM/N_PAIRS/N_EQ, GROUP, set_group, group_mul_np,
    build_tables  : repro/train_d57_abeq.py lines 32-74 (verbatim);
  - make_split   : repro/train_d57_abeq.py main() lines 380-384, promoted
    into a function (identical RNG construction: SeedSequence(seed) ->
    default_rng -> permutation; unsorted id_train per the §6cz-prime
    lesson -- np.sort is used ONLY for sorted bookkeeping by callers).
  - M_NORM/D_NORM: train_d57_abeq.py lines 37-41 (sweep-only; kept for
    fidelity, unused by the zp mainline).

WARNING (load-bearing): N_ELEM/N_PAIRS/N_EQ are module globals mutated by
set_group.  D57Model's DEFAULT ARGUMENT n_elem=N_ELEM is bound at class
definition time in model.py -- do NOT try to "fix" this to late binding;
the frozen 114 binding is what the archived bridge checkpoints were built
with (see model.py docstring).
"""
import numpy as np

N_ROT = 57
N_ELEM = 114
N_PAIRS = N_ELEM * N_ELEM
N_EQ = 114                 # extra vocabulary entry for the "=" token

M_NORM = 2.0
D_NORM = 2.0

GROUP = {'name': 'd57', 'n': 114, 'p': 113}


def set_group(name, p=113):
    """train_d57_abeq.py lines 47-52 (verbatim)."""
    global N_ELEM, N_EQ, N_PAIRS, GROUP
    GROUP = {'name': name, 'n': (p if name == 'zp' else 114), 'p': p}
    N_ELEM = GROUP['n']
    N_EQ = GROUP['n']
    N_PAIRS = N_ELEM * N_ELEM


def group_mul_np(a, b):
    """D57 (dihedral) or Z_p (cyclic addition) depending on GROUP.
    train_d57_abeq.py lines 55-66 (verbatim)."""
    if GROUP['name'] == 'zp':
        return (a + b) % GROUP['p']
    eps_a = a // N_ROT
    i_a = a % N_ROT
    eps_b = b // N_ROT
    j_b = b % N_ROT
    sign = 1 - 2 * eps_a
    i_out = (i_a + sign * j_b) % N_ROT
    eps_out = (eps_a + eps_b) % 2
    return eps_out * N_ROT + i_out


def build_tables():
    """train_d57_abeq.py lines 69-74 (verbatim)."""
    n = GROUP['n']
    all_a = np.arange(n).reshape(1, -1).repeat(n, axis=0).ravel()
    all_b = np.arange(n).reshape(-1, 1).repeat(n, axis=1).ravel()
    all_y = group_mul_np(all_a, all_b)
    return all_a.astype(np.int64), all_b.astype(np.int64), all_y.astype(np.int64)


def make_split(seed, split_frac=0.30):
    """Train/val pair-index split.  Construction lifted verbatim from
    train_d57_abeq.py main() lines 380-384 (and identical to
    r92_gate_sampler.build_train lines 67-74).

    Returns (id_train UNSORTED, va_idx).  The unsorted id_train is the
    harness reference -- batch<->pair correspondence code must consume
    it unsorted (§6cz-prime: sorted/unsorted mismatch was the R87
    confound; np.sort is only for sorted bookkeeping).
    """
    ss = np.random.SeedSequence(seed)
    rng = np.random.default_rng(ss)
    perm = rng.permutation(N_PAIRS)
    n_train = int(round(N_PAIRS * split_frac))
    return perm[:n_train], perm[n_train:]


def tensors_from_split(all_a, all_b, all_y, id_train, va_idx):
    """torch int64 tensors per split.  train_d57_abeq.py lines 386-391
    (same from_numpy(...).long() calls, same order)."""
    import torch
    tr_a = torch.from_numpy(all_a[id_train]).long()
    tr_b = torch.from_numpy(all_b[id_train]).long()
    tr_y = torch.from_numpy(all_y[id_train]).long()
    va_a = torch.from_numpy(all_a[va_idx]).long()
    va_b = torch.from_numpy(all_b[va_idx]).long()
    va_y = torch.from_numpy(all_y[va_idx]).long()
    return tr_a, tr_b, tr_y, va_a, va_b, va_y
