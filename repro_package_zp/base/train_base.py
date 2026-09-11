"""base/train_base.py — canonical base-training entry (package-owned).

Provenance (M2, 2026-09-02): training loop transcribed verbatim from
repro/train_d57_abeq.py main() lines 368-598, restricted to the zp
bridge configs, built ONLY on common/ modules (no repro/ import).
Op order, RNG consumption order, and eval cadence are identical; any
deviation breaks bit-exactness with the archive.

Modes:
  --verify-summary PATH : replay one (family, seed) run in memory and
      compare bit-exact against the archived bridge summary.json
      (history / first_cross / final / od / n_params / n_train).  This
      is the M2 reconciliation for the extraction; GPU required
      (archived runs were CUDA; CPU replay will NOT match bitwise).
  (10-seed B0 driver: DEFERRED per user 2026-09-02 -- do not run batch
   retrains until explicitly approved.  The per-run function below is
   the only sanctioned entry; emission follows base/README SNAPSHOT_SPEC.)
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from common import task
from common.model import D57Model, make_optimizer, accuracy, od_accuracy


def run_base(seed, family='A', steps=30000, eval_every=2000, batch=512,
             lr=1e-3, wd=1.0, split_frac=0.30, ln_eps=1e-5, eq_alpha=0.0,
             snap_at=(), out_dir=None, dev_str=None):
    """One canonical base run.  family: A(natS) C(fqk200) B(noeqemb) D(fwv200).

    train_d57_abeq.py main() transcription notes (load-bearing for
    bit-exactness):
      * np.random.seed(seed) + torch.manual_seed(seed) BEFORE model init
        (lines 371-372); global CPU RNG feeds one torch.randint per step
        (line 521) -- nothing else may draw from it;
      * model constructed on CPU then .to(dev) (line 393-394);
      * eval block runs after opt.step() within the same iteration and
        records the current step's loss (lines 574-580);
      * first-crossing recorded at eval steps only, va_acc > 0.95
        (lines 581-585).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = torch.device(dev_str or ('cuda' if torch.cuda.is_available() else 'cpu'))

    task.set_group('zp', 113)
    all_a, all_b, all_y = task.build_tables()
    id_train, va_idx = task.make_split(seed, split_frac)
    (tr_a, tr_b, tr_y,
     va_a, va_b, va_y) = task.tensors_from_split(all_a, all_b, all_y,
                                                 id_train, va_idx)

    arch = 'abeq_noeqemb' if family == 'B' else 'abeq'
    fqk = 200 if family == 'C' else 0
    fwv = 200 if family == 'D' else 0
    fp2 = 0                           # TAG TRAP: bridge tag token "fp20" is
                                      # prefix "fp2" + value 0, i.e. NO p2
                                      # freezing (freeze_p2_steps=0); first
                                      # M2 replay run falsely used 20 and
                                      # diverged.  Verified against archive
                                      # filenames on 2026-09-02.

    model = D57Model(pos_mode='zeros', arch=arch, ln_eps=ln_eps,
                     eq_alpha=eq_alpha).to(dev)
    opt, decay, no_decay = make_optimizer(model, 'wd_0011', lr=lr, wd=wd)

    loss_fn = nn.CrossEntropyLoss()
    first_cross_step = None
    best_val, best_step = 0.0, 0
    history = []
    qk_trace = None   # populated over steps 1..200 (SNAPSHOT_SPEC patch)

    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(tr_a.shape[0], (batch,))
        a, b, y = tr_a[idx].to(dev), tr_b[idx].to(dev), tr_y[idx].to(dev)
        opt.zero_grad()
        loss = loss_fn(model(a, b), y)
        loss.backward()
        if step <= fqk:
            model.Wq.weight.grad = None
            model.Wk.weight.grad = None
        if step <= fwv:
            model.Wv.weight.grad = None
        if step <= fp2 and model.pos.grad is not None:
            model.pos.grad[2].zero_()
        opt.step()

        if step % eval_every == 0 or step == 1:
            tr_acc = accuracy(model, tr_a, tr_b, tr_y, dev)
            va_acc = accuracy(model, va_a, va_b, va_y, dev)
            history.append({"step": step, "train_acc": tr_acc,
                            "val_acc": va_acc, "loss": loss.item()})
            if first_cross_step is None and va_acc > 0.95:
                first_cross_step = step
                # SNAPSHOT_SPEC (10-seed patch 2026-09-05): firstcross.pt
                # -- full model+opt at the first crossing eval step.
                if out_dir is not None:
                    os.makedirs(out_dir, exist_ok=True)
                    base = (f"seed{seed}_{arch}_a{eq_alpha}_eps{ln_eps}"
                            f"_fqk{fqk}_fwv{fwv}_fmlp0_fp2{fp2}"
                            f"_zeros_wd_0011")
                    torch.save({"step": step, "model": model.state_dict(),
                                "opt": opt.state_dict(), "val_acc": va_acc},
                               os.path.join(out_dir,
                                            f"{base}_firstcross.pt"))
            if va_acc > best_val:
                best_val, best_step = va_acc, step

        # SNAPSHOT_SPEC (10-seed patch 2026-09-05): qk_trace_0_200.pt --
        # per-step Wq/Wk parameter flow over the first 200 steps
        # (required input for K05 formation accounting; new artifact,
        # absent from the archive by design).
        if step <= 200 and out_dir is not None:
            if qk_trace is None:
                qk_trace = {"steps": [], "Wq": [], "Wk": []}
            qk_trace["steps"].append(step)
            qk_trace["Wq"].append(
                model.Wq.weight.detach().cpu().clone())
            qk_trace["Wk"].append(
                model.Wk.weight.detach().cpu().clone())

        if step in snap_at and out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            base = (f"seed{seed}_{arch}_a{eq_alpha}_eps{ln_eps}"
                    f"_fqk{fqk}_fwv{fwv}_fmlp0_fp2{fp2}_zeros_wd_0011")
            torch.save({"step": step, "model": model.state_dict(),
                        "opt": opt.state_dict(), "val_acc": va_acc},
                       os.path.join(out_dir, f"{base}_full{step:07d}.pt"))

    result = {
        "seed": seed, "family": family, "pos_mode": "zeros",
        "wd_arm": "wd_0011", "n_params": sum(p.numel()
                                             for p in model.parameters()),
        "split": split_frac, "n_train": int(tr_a.shape[0]),
        "n_val": int(va_a.shape[0]), "group": "zp", "p": 113,
        "history": history, "optimizer": "adamw",
        "first_cross_step": first_cross_step, "final_step": steps,
        "final_val_acc": history[-1]["val_acc"] if history else None,
        "final_od_acc": od_accuracy(model, va_a, va_b, va_y, dev),
        "best_val_acc": best_val, "best_val_step": best_step,
    }
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        base = (f"seed{seed}_{arch}_a{eq_alpha}_eps{ln_eps}"
                f"_fqk{fqk}_fwv{fwv}_fmlp0_fp2{fp2}_zeros_wd_0011")
        torch.save({"step": steps, "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "val_acc": result["final_val_acc"]},
                   os.path.join(out_dir, f"{base}_final.pt"))
        # SNAPSHOT_SPEC (10-seed patch): qk_trace + summary sha256
        if qk_trace is not None:
            torch.save(qk_trace,
                       os.path.join(out_dir, f"{base}_qk_trace_0_200.pt"))
        import hashlib
        with open(os.path.join(out_dir, f"{base}_summary.json"),
                  "w") as f:
            json.dump(result, f, indent=2)
        h = hashlib.sha256()
        with open(os.path.join(out_dir, f"{base}_summary.json"),
                  "rb") as f:
            h.update(f.read())
        result["summary_sha256"] = h.hexdigest()
        with open(os.path.join(out_dir, f"{base}_summary.json"),
                  "w") as f:
            json.dump(result, f, indent=2)
    return result


def verify_against_summary(result, summary_path):
    """Bit-exact comparison against an archived bridge summary.json.
    history values must match EXACTLY (==); JSON round-trips preserve
    float repr, so any mismatch is a real extraction divergence."""
    with open(summary_path) as f:
        s = json.load(f)
    checks = []
    for key in ("n_params", "n_train", "n_val", "split", "seed"):
        checks.append((key, result[key] == s[key], result[key], s[key]))
    checks.append(("len(history)", len(result["history"]) == len(s["history"]),
                   len(result["history"]), len(s["history"])))
    bad = 0
    for mine, ref in zip(result["history"], s["history"]):
        for k in ("step", "train_acc", "val_acc", "loss"):
            if mine[k] != ref[k]:
                bad += 1
                if bad <= 5:
                    print(f"  history MISMATCH at step {ref['step']} "
                          f"{k}: mine={mine[k]!r} ref={ref[k]!r}")
    checks.append(("history values", bad == 0, f"{bad} mismatches",
                   "0 mismatches"))
    for key in ("first_cross_step", "final_val_acc", "final_od_acc",
                "best_val_acc", "best_val_step"):
        mine, ref = result[key], s[key]
        if key == "final_od_acc" and mine != mine and ref == ref:
            # KNOWN LEGACY DIVERGENCE (documented, not a failure):
            # zp is commutative, so the OD subset (a.b != b.a) is EMPTY and
            # od_accuracy correctly returns nan.  The archive predates the
            # od_accuracy fix (its own docstring: "an earlier version ...
            # silently applied the dihedral law to Z_p runs"), hence a
            # non-nan value computed under a different (wrong-for-zp)
            # order law.  Training trajectory bit-exactness is judged by
            # history/first_cross/final/best, which are unaffected.
            print(f"  ok  {key}: mine=nan (correct for zp) vs "
                  f"ref={ref} (legacy od definition in archive)")
            continue
        if isinstance(mine, float) and isinstance(ref, float)                 and mine != mine and ref != ref:
            # both nan (zp OD subset is empty on both sides) -- equal
            checks.append((key, True, "nan", "nan"))
            continue
        checks.append((key, mine == ref, mine, ref))
    ok = all(c[1] for c in checks)
    print(f"[verify] {os.path.basename(summary_path)}: "
          f"{'PASS (bit-exact)' if ok else 'FAIL'}")
    for name, passed, mine, ref in checks:
        mark = 'ok ' if passed else 'FAIL'
        if not passed or name in ("n_params", "first_cross_step"):
            print(f"  {mark} {name}: mine={mine} ref={ref}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["A", "C", "B", "D"], default="A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--verify-summary", default="",
                    help="archive summary.json to reconcile against")
    ap.add_argument("--out", default="", help="emit checkpoints here")
    args = ap.parse_args()
    res = run_base(args.seed, args.family, steps=args.steps,
                   eval_every=args.eval_every,
                   snap_at=(200, 700), out_dir=args.out or None)
    if args.verify_summary:
        ok = verify_against_summary(res, args.verify_summary)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
