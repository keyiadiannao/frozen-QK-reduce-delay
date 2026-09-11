"""common/model.py — one-layer abeq Transformer (verbatim extraction).

Provenance (M2 extraction 2026-09-02): repro/train_d57_abeq.py lines
77-257 (sinusoidal_table, apply_rope, D57Model), copied verbatim.

LOAD-BEARING FROZEN BINDING -- read before "cleaning up":
  D57Model.__init__ has default n_elem=N_ELEM.  Python binds default
  arguments at class-definition time, so the default is FROZEN at 114
  (the d57 import-time value) even after task.set_group('zp', 113).
  Every archived bridge checkpoint (bridge_zp_*) was built this way:
    n_params = 227826 =
      emb 115x128 (14720) + pos 3x128 (384) + LN 3x256 (768)
      + Wq/Wk/Wv/Wo 4x16384 (65536) + mlp1 (66048) + mlp2 (65664)
      + Wu 114 outputs (14706)
  Changing the default to late binding would change the parameter
  shapes, break checkpoint load_state_dict, and silently invalidate
  the entire archive.  Do not touch.  (Same fact documented in
  train_d57_abeq.fourier_metrics docstring, lines 490-500.)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import task

N_ELEM_AT_DEF = task.N_ELEM   # 114 -- the value bound into defaults below


def sinusoidal_table(n_pos, d_model):
    """train_d57_abeq.py lines 77-85 (verbatim)."""
    pe = torch.zeros(n_pos, d_model)
    position = torch.arange(n_pos).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float()
                         * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def apply_rope(x, base=10000.0):
    """Rotary position embedding.  x: (B, n_heads, n_pos, d_head).
    train_d57_abeq.py lines 88-107 (verbatim)."""
    P, D = x.shape[2], x.shape[3]
    i = torch.arange(0, D, 2, device=x.device, dtype=x.dtype)
    theta = 1.0 / (base ** (i / D))
    t = torch.arange(P, device=x.device, dtype=x.dtype)
    ang = torch.outer(t, theta)                       # (P, D/2)
    cos, sin = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    o1 = x1 * cos[None, None] - x2 * sin[None, None]
    o2 = x1 * sin[None, None] + x2 * cos[None, None]
    out = torch.empty_like(x)
    out[..., 0::2] = o1
    out[..., 1::2] = o2
    return out


class D57Model(nn.Module):
    """train_d57_abeq.py lines 110-257 (verbatim, comments preserved)."""
    def __init__(self, d_model=128, n_heads=4, d_ff=512, n_elem=N_ELEM_AT_DEF,
                 pos_mode='zeros', alpha=1.0, arch='abeq', ln_eps=1e-5,
                 eq_alpha=0.0):
        super().__init__()
        self.pos_mode = pos_mode
        self.arch = arch
        self.eq_alpha = eq_alpha
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.n_elem_emb = n_elem + 1
        self.emb = nn.Embedding(self.n_elem_emb, d_model)

        if arch in ('ab_read0', 'ab_pool'):
            self.pos = nn.Parameter(torch.zeros(2, d_model))
        elif pos_mode == 'zeros':
            self.pos = nn.Parameter(torch.zeros(3, d_model))
        elif pos_mode == 'rand':
            self.pos = nn.Parameter(torch.randn(3, d_model) * 0.02)
        elif pos_mode == 'sweep':
            g = torch.Generator().manual_seed(7777)
            m = torch.randn(d_model, generator=g)
            m = m / m.norm() * task.M_NORM
            d = torch.randn(d_model, generator=g)
            d = d / d.norm() * task.D_NORM
            self.register_buffer('pos', torch.stack([m + alpha * d,
                                                     m - alpha * d]))
        elif pos_mode == 'sin':
            self.register_buffer('pos', sinusoidal_table(2, d_model))
        elif pos_mode == 'sinc':
            t = sinusoidal_table(2, d_model)
            self.register_buffer('pos', t - t.mean(0, keepdim=True))
        elif pos_mode == 'frz':
            self.register_buffer('pos', torch.randn(2, d_model) * 0.25)
        elif pos_mode == 'rope':
            self.register_buffer('pos', torch.zeros(2, d_model))
        elif pos_mode == 'none':
            self.register_buffer('pos', torch.zeros(2, d_model))
        else:
            raise ValueError(pos_mode)
        self.rope = (pos_mode == 'rope')

        self.ln1 = nn.LayerNorm(d_model, eps=ln_eps)
        if arch == 'abeq_fixedanchor':
            g = torch.Generator().manual_seed(1234)
            u = torch.randn(d_model, generator=g)
            u = u / u.norm() * math.sqrt(d_model)
            self.register_buffer('anchor_u', u)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln_f = nn.LayerNorm(d_model)

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        self.mlp1 = nn.Linear(d_model, d_ff, bias=True)
        self.mlp2 = nn.Linear(d_ff, d_model, bias=True)
        self.Wu = nn.Linear(d_model, n_elem, bias=True)

    def forward(self, a, b):
        if self.arch == 'ab_pool':
            B = a.shape[0]
            tok = torch.stack([a, b], dim=1)
            x = self.emb(tok) + self.pos[None, :, :]
            h = self.ln1(x)
            q = self.Wq(h).view(B, 2, self.n_heads, self.d_head).transpose(1, 2)
            k = self.Wk(h).view(B, 2, self.n_heads, self.d_head).transpose(1, 2)
            v = self.Wv(h).view(B, 2, self.n_heads, self.d_head).transpose(1, 2)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
            attn = scores.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).contiguous().view(B, 2, self.d_model)
            x = x + self.Wo(out)
            h2 = self.ln2(x)
            x = x + self.mlp2(F.gelu(self.mlp1(h2)))
            return self.Wu(self.ln_f(x.mean(dim=1)))
        if self.arch == 'ab_read0':
            B = a.shape[0]
            tok = torch.stack([a, b], dim=1)
            x = self.emb(tok) + self.pos[None, :, :]
            h = self.ln1(x)
            q = self.Wq(h).view(B, 2, self.n_heads, self.d_head).transpose(1, 2)
            k = self.Wk(h).view(B, 2, self.n_heads, self.d_head).transpose(1, 2)
            v = self.Wv(h).view(B, 2, self.n_heads, self.d_head).transpose(1, 2)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
            attn = scores.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).contiguous().view(B, 2, self.d_model)
            x = x + self.Wo(out)
            h2 = self.ln2(x)
            x = x + self.mlp2(F.gelu(self.mlp1(h2)))
            return self.Wu(self.ln_f(x[:, 0, :]))
        B = a.shape[0]
        eq = torch.full_like(a, task.N_EQ)
        tok = torch.stack([a, b, eq], dim=1)          # (B,3)
        x = self.emb(tok) + self.pos[None, :, :]      # 3 learned positions
        if self.arch == 'abeq_noeqemb':
            x = torch.cat([x[:, :2, :], self.pos[None, 2, :].expand(B, 1, -1)],
                          dim=1)   # '=' carries NO token embedding, only p_2
        elif self.arch == 'abeq_fixedanchor':
            third = self.pos[None, 2, :] + self.eq_alpha * self.anchor_u[None, :]
            x = torch.cat([x[:, :2, :], third.expand(B, 1, -1)], dim=1)

        h = self.ln1(x)
        q = self.Wq(h).view(B, 3, self.n_heads, self.d_head).transpose(1, 2)
        k = self.Wk(h).view(B, 3, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(h).view(B, 3, self.n_heads, self.d_head).transpose(1, 2)
        if self.rope:
            q, k = apply_rope(q), apply_rope(k)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, 3, self.d_model)
        x = x + self.Wo(out)

        h2 = self.ln2(x)
        m = self.mlp2(F.gelu(self.mlp1(h2)))
        x = x + m

        last = x[:, 2, :]                              # privileged readout node
        return self.Wu(self.ln_f(last))


def make_optimizer(model, wd_arm, lr=1e-3, wd=1.0):
    """train_d57_abeq.py lines 260-276 (verbatim)."""
    decay, no_decay = [], []
    attn_w = {"Wq", "Wk", "Wv", "Wo"}
    for name, p in model.named_parameters():
        if name == "pos":
            no_decay.append(p)
            continue
        is_weight = "weight" in name
        # 2026-09-04 audit fix: names are "<module>.weight" so the old
        # endswith(".Wq") never matched -> wd_1111 silently equalled
        # wd_0011.  Prefix check is correct; wd_0011 grouping unchanged
        # (verified bitwise on the first-step reference).
        is_attn = any(name.startswith(w + ".") for w in attn_w)
        if is_weight and (name.startswith("mlp") or name.startswith("Wu")
                          or (wd_arm == "wd_1111" and is_attn)):
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [{"params": decay, "weight_decay": wd},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999)), decay, no_decay


@torch.no_grad()
def accuracy(model, a, b, y, dev, bs=4096):
    """train_d57_abeq.py lines 279-289 (verbatim)."""
    model.eval()
    correct = 0
    n = a.shape[0]
    for s in range(0, n, bs):
        aa = a[s:s + bs].to(dev)
        bb = b[s:s + bs].to(dev)
        logits = model(aa, bb)
        correct += (logits.argmax(dim=-1).cpu() == y[s:s + bs]).sum().item()
    return correct / n


@torch.no_grad()
def od_accuracy(model, a, b, y, dev, bs=4096):
    """train_d57_abeq.py lines 292-305 (verbatim)."""
    swap = torch.from_numpy(task.group_mul_np(b.numpy(), a.numpy())).long()
    od = (y != swap).numpy()
    if od.sum() == 0:
        return float('nan')
    return accuracy(model, a[od], b[od], y[od], dev, bs)
