"""K01 run.py — necessity 2x2 at t=200 (fate lock in the non-QK subsystem).

Package port of repro/amp_necess2x2.py (M2.5, 2026-09-02).  Verbatim
except: bridge paths via common.states (state codes F/S -> families
C/A), val_probe from common.readouts, claim-dir output.

2x2: state in {F200, S200} x post-200 QK in {trainable, frozen}; all
arms replay from the same theta_200 snapshots with their own opt state
and the main-stream RNG chain (seed -> model init -> discard the 200
frozen-phase draws -> continue); CAP=4000, vacc>0.9 crossing.

Claim (ledger K1): by t=200 the slow fate is already written into the
non-QK subsystem -- freezing QK 200->4000 leaves va~0.30 uncrossed
(S-frozen) while F-frozen crosses within the F-train window (J1/J4
pattern, seed-paired).  Preregistered Round 51; STATE_SPACE Assay A.
"""
import os
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
REPRO_DIR = os.environ.get('REPRO_DIR', os.path.join(
    os.path.dirname(_PKG_ROOT), 'repro'))

from common import task, states
from common.model import D57Model, make_optimizer
from common.readouts import ZPReadouts

task.set_group('zp', 113)
DEV = 'cuda'
LOSS = nn.CrossEntropyLoss()
BLOCKS = ["emb.weight", "pos", "Wq.weight", "Wk.weight", "Wv.weight",
          "Wo.weight", "mlp1.weight", "mlp1.bias", "mlp2.weight",
          "mlp2.bias", "Wu.weight", "Wu.bias"]
NONQK = [n for n in BLOCKS if n not in ("Wq.weight", "Wk.weight")]
QK = ["Wq.weight", "Wk.weight"]
SNAP_EVERY = 25
CAP = 4000
FAMILY = {'S': 'A', 'F': 'C'}   # state codes -> bridge family letters

R = ZPReadouts(DEV)


def ckpt(state, seed):
    return states.bridge_ckpt_path(REPRO_DIR, FAMILY[state], seed, None,
                                   'full0000200')


def load_state(src, seed):
    torch.manual_seed(seed)
    m = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    sd = torch.load(src, map_location='cpu', weights_only=False)
    m.load_state_dict(sd['model'])
    opt, _, _ = make_optimizer(m, 'wd_0011')
    opt.load_state_dict(sd['opt'])
    return m, opt


def run_arm(state, mode, seed):
    """Replay from theta_200 with post-200 QK trainable or frozen."""
    va = R.val_probe(seed)
    m, opt = load_state(ckpt(state, seed), seed)
    # main-stream RNG chain continuation (amp3_brun structure)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    perm = rng.permutation(task.N_PAIRS)
    tr_a = torch.from_numpy(R.ALL_A[perm[:R.N_TRAIN]]).long()
    tr_b = torch.from_numpy(R.ALL_B[perm[:R.N_TRAIN]]).long()
    tr_y = torch.from_numpy(R.ALL_Y[perm[:R.N_TRAIN]]).long()
    for _ in range(200):
        torch.randint(tr_a.shape[0], (512,))

    rec = {'t': [], 'val_acc': [], 'loss': [], 'gnorm': {},
           'gvalc': [], 'gvalqk': []}
    crossed = None
    for t in range(1, CAP + 1):
        idx = torch.randint(tr_a.shape[0], (512,))
        m.train()
        opt.zero_grad()
        a = tr_a[idx].to(DEV)
        b = tr_b[idx].to(DEV)
        y = tr_y[idx].to(DEV)
        LOSS(m(a, b), y).backward()
        if mode == 'frozen':
            m.Wq.weight.grad = None
            m.Wk.weight.grad = None
        opt.step()
        if t % SNAP_EVERY == 0 or t == CAP:
            m.eval()
            with torch.no_grad():
                logits = m(va[0], va[1])
                vacc = float((logits.argmax(-1) == va[2]).float().mean())
                vloss = float(LOSS(logits, va[2]).item())
            m.zero_grad()
            LOSS(m(va[0], va[1]), va[2]).backward()
            gn = {}
            for n, p in m.named_parameters():
                if n in BLOCKS and p.grad is not None:
                    gn[n] = float(p.grad.norm().item())
            rec['t'].append(t)
            rec['val_acc'].append(vacc)
            rec['loss'].append(vloss)
            rec['gnorm'][t] = gn
            rec['gvalc'].append(float(np.sqrt(
                sum(gn[n] ** 2 for n in NONQK))))
            rec['gvalqk'].append(float(np.sqrt(
                sum(gn[n] ** 2 for n in QK))))
            if vacc > 0.9:
                crossed = t
                break
    rec['crossed'] = crossed
    return rec


def main():
    seeds = tuple(int(s) for s in os.environ.get(
        'CLAIM_SEEDS', '0,1,2').split(','))
    out = {}
    for state in ('F', 'S'):
        for mode in ('train', 'frozen'):
            for seed in seeds:
                key = (state, mode, seed)
                out[key] = run_arm(state, mode, seed)
                r = out[key]
                print(f"{state}-{mode} seed{seed}: crossed={r['crossed']} "
                      f"last_t={r['t'][-1]} "
                      f"final_va={r['val_acc'][-1]:.3f} "
                      f"gC {r['gvalc'][0]:.2f}->{max(r['gvalc']):.2f}",
                      flush=True)

    print('\n=== crossing table (replay steps post-theta200, cap %d) ==='
          % CAP)
    for state in ('F', 'S'):
        for mode in ('train', 'frozen'):
            cs = [out[(state, mode, s)]['crossed'] for s in seeds]
            print(f"  {state}-{mode:6s}: {cs}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({f"{s}-{m}-{d}": r for (s, m, d), r in out.items()},
                    f)
    print('\nsaved results.pkl')


if __name__ == '__main__':
    main()
