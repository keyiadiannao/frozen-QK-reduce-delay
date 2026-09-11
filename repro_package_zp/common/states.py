"""common/states.py — checkpoint I/O, state transplant, deterministic fork.

Provenance (M2 extraction 2026-09-02):
  - save_full / ckpt dict format : train_d57_abeq.py lines 515-517
    ({"step", "model", "opt", "val_acc"}; snap files omit "opt");
  - _src_sd / block transplant   : train_d57_abeq.py main() lines 421-467
    (QK block = {Wq.weight, Wk.weight});
  - fork deepcopy                : R94 batch-chain fork machinery
    (EXTENSION_SUMMARY §6de/§6df; R106 smoke lesson: opt.state_dict()
    returns LIVE references -> must deepcopy before storing/loading a
    second fork, else the final-moments fork NaNs).

Bridge checkpoint layout (repro/bridge_zp_*, produced by
train_d57_abeq.py; tag = f"seed{seed}_{arch}_a{eq_alpha}_eps{ln_eps}
_fqk{freeze_qk}_fwv{freeze_wv}_fmlp{freeze_mlp}_fp2{freeze_p2}_​{pos}_{wd_arm}"):
  bridge_zp_A = S  (native, fqk0)      witness: full0000200
  bridge_zp_C = F  (fqk200)            witness: full0000200, release
  bridge_zp_B = Z  (noeqemb, fqk0)     witness: full0000200
  bridge_zp_D = fwv200 control         witness: full0000200
"""
import copy
import glob
import os

import torch

QK_NAMES = {"Wq.weight", "Wk.weight"}


def load_ckpt(path):
    """train_d57_abeq.py main() _src_sd (lines 421-424)."""
    sd0 = torch.load(path, map_location='cpu', weights_only=False)
    return sd0['model'] if 'model' in sd0 else sd0, \
        (sd0.get('opt') if isinstance(sd0, dict) else None)


def load_model(model, path):
    """Load weights-only into an existing model (weights fresh, opt fresh).
    train_d57_abeq.py lines 395-398."""
    sd, _ = load_ckpt(path)
    model.load_state_dict(sd)
    return model


def transplant_blocks(model, nonqk_path='', qk_path=''):
    """Block-wise weight transplant (weights only; moments handled by the
    training script when needed).  train_d57_abeq.py lines 426-445.
    nonqk_path fills every param EXCEPT QK; qk_path fills QK only."""
    dst = model.state_dict()
    merged = {}
    for path, which in ((nonqk_path, 'nonqk'), (qk_path, 'qk')):
        if not path:
            continue
        src, _ = load_ckpt(path)
        for name, ten in src.items():
            in_qk = name in QK_NAMES
            if (which == 'qk') == in_qk and name in dst:
                merged[name] = ten.clone()
    missing = set(dst) - set(merged)
    if merged and missing:
        raise RuntimeError(
            f"transplant left params unloaded: {sorted(missing)}")
    if merged:
        model.load_state_dict(merged, strict=True)
    return model


def fork_state(model, optimizer):
    """Deterministic fork snapshot: DEEPCOPY both model and optimizer
    state (opt.state_dict() tensors are live references -- R106 smoke-2
    lesson; shallow copies alias the training optimizer's moments)."""
    sd = copy.deepcopy(model.state_dict())
    opt_sd = copy.deepcopy(optimizer.state_dict())
    return sd, opt_sd


def bridge_ckpt_path(repro_dir, family, seed, config, snap):
    """Resolve a bridge checkpoint path by (family, seed, config, snap).

    family in {A,C,B,D}; config in {'S','F','Z','WV'} mapping:
      A/S : seed{N}_abeq_a0.0_eps1e-05_fqk0_fwv0_fmlp0_fp20_zeros_wd_0011
      C/F : ..._fqk200_...
      B/Z : seed{N}_abeq_noeqemb_a0.0_eps1e-05_fqk0_...
      D/WV: ..._fqk0_fwv200_...
    snap in {'full0000200','full0000700','final','firstcross','release'}.
    """
    base = {
        'A': 'seed{seed}_abeq_a0.0_eps1e-05_fqk0_fwv0_fmlp0_fp20_zeros_wd_0011',
        'C': 'seed{seed}_abeq_a0.0_eps1e-05_fqk200_fwv0_fmlp0_fp20_zeros_wd_0011',
        'B': 'seed{seed}_abeq_noeqemb_a0.0_eps1e-05_fqk0_fwv0_fmlp0_fp20_zeros_wd_0011',
        'D': 'seed{seed}_abeq_a0.0_eps1e-05_fqk0_fwv200_fmlp0_fp20_zeros_wd_0011',
    }[family].format(seed=seed)
    pat = os.path.join(repro_dir, f'bridge_zp_{family}', f'{base}_{snap}.pt')
    hits = sorted(glob.glob(pat))
    if len(hits) != 1:
        raise FileNotFoundError(f'{pat} -> {len(hits)} hits')
    return hits[0]
