"""common/readouts.py — shared Zp readout machinery (verbatim extraction).

Provenance (M2.5, 2026-09-02): consolidated from r103/r104 (identical
code in both; r103_frozen_carrier_retention.py lines 57-129/209-218 and
r104_native_addremove.py lines 60-129/291-299).  Relocated verbatim:
the RNG constructions (SeedSequence(seed) split, default_rng(555) val
probe, SeedSequence([9000+seed]) batch chains) are load-bearing for
bit-exact reconciliation with the archived pkls.

Users: K06 (R104 add/remove), K07 (R103 retention), K11/K13 lineage.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from common import task


class ZPReadouts:
    def __init__(self, dev):
        task.set_group('zp', 113)
        self.dev = dev
        self.P = task.GROUP['p']
        self.N_PAIRS = task.N_PAIRS
        self.N_TRAIN = int(round(self.N_PAIRS * 0.30))
        self.ALL_A, self.ALL_B, self.ALL_Y = task.build_tables()
        Y_ARR = self.ALL_Y
        self.CLS = {c: np.where(Y_ARR == c)[0] for c in range(self.P)}
        assert len(self.CLS[0]) == self.P   # each sum class has P pairs

    def val_probe(self, seed):
        """r103/r104 val_probe (verbatim)."""
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        perm = rng.permutation(task.N_PAIRS)
        rngv = np.random.default_rng(555)
        vb = rngv.permutation(perm[self.N_TRAIN:])[:1024]
        dev = self.dev
        return (torch.from_numpy(self.ALL_A[vb]).long().to(dev),
                torch.from_numpy(self.ALL_B[vb]).long().to(dev),
                torch.from_numpy(self.ALL_Y[vb]).long().to(dev))

    def plain_forward_attn(self, m, a, b):
        """r103/r104 plain_forward_attn (verbatim)."""
        B = a.shape[0]
        eq = torch.full_like(a, task.N_EQ)
        tok = torch.stack([a, b, eq], dim=1)
        x = m.emb(tok) + m.pos[None, :, :]
        h = m.ln1(x)
        q = m.Wq(h).view(B, 3, m.n_heads, m.d_head).transpose(1, 2)
        k = m.Wk(h).view(B, 3, m.n_heads, m.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(m.d_head)
        attn = scores.softmax(dim=-1)
        v = m.Wv(h).view(B, 3, m.n_heads, m.d_head).transpose(1, 2)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, 3, m.d_model)
        x = x + m.Wo(out)
        h2 = m.ln2(x)
        mm = m.mlp2(F.gelu(m.mlp1(h2)))
        x = x + mm
        last = x[:, 2, :]
        return m.Wu(m.ln_f(last)), attn

    @torch.no_grad()
    def own_codes(self, m, eps=1e-8):
        """CLR routing codes over the full grid (r103 own_codes)."""
        P, N_PAIRS, dev = self.P, self.N_PAIRS, self.dev
        tid = torch.arange(N_PAIRS, dtype=torch.long, device=dev)
        a, b = tid % P, tid // P
        outs = []
        for s in range(0, N_PAIRS, 4096):
            _, attn = self.plain_forward_attn(m, a[s:s + 4096],
                                              b[s:s + 4096])
            a2 = attn[:, :, 2, :].double().clamp_min(eps)
            c = torch.log(a2)
            c = c - c.mean(dim=-1, keepdim=True)
            outs.append(c.reshape(c.shape[0], -1).float().cpu())
        return torch.cat(outs, 0).numpy()

    def retrieve_self(self, query, ref):
        """Own-snapshot cross-time self-retrieval (r103 retrieve_self)."""
        P, N_PAIRS, CLS = self.P, self.N_PAIRS, self.CLS
        Q = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-30)
        R = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-30)
        hits = 0
        for c in range(P):
            idx = CLS[c]
            sims = Q[idx] @ R[idx].T
            hits += int((sims.argmax(1) ==
                         np.arange(len(idx))).sum())
        return hits / N_PAIRS

    def margin_self(self, query, ref):
        """m_i = s(i,i) - max_{j!=i, same-sum} s(i,j); returns
        (median, mean) over all pairs (r103 margin_self)."""
        P, CLS = self.P, self.CLS
        Q = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-30)
        R = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-30)
        margins = []
        for c in range(P):
            idx = CLS[c]
            sims = Q[idx] @ R[idx].T
            d = np.diag(sims)
            off = sims.copy()
            np.fill_diagonal(off, -np.inf)
            margins.append(d - off.max(1))
        margins = np.concatenate(margins)
        return float(np.median(margins)), float(margins.mean())

    def batch_chain(self, seed, steps, batch=512):
        """(id_train UNSORTED, explicit chain) -- r103/r104 construction.
        Chains are identical across arms by construction."""
        rc = np.random.default_rng(np.random.SeedSequence([9000 + seed]))
        chain = [rc.integers(0, self.N_TRAIN, batch) for _ in range(steps)]
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        perm = rng.permutation(task.N_PAIRS)
        id_train = perm[:self.N_TRAIN]       # UNSORTED (§6cz-prime)
        return id_train, chain
