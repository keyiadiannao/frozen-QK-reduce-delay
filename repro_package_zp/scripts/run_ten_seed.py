"""scripts/run_ten_seed.py — one-command driver for the 10-seed
confirmatory (Stages 0-7 of TEN_SEED_RUN_PLAN.md).

Usage (server):
    python scripts/run_ten_seed.py --repro-dir /path/bridges \
        --out /path/tenseed_out [--stage N] [--seeds 0,...,9]

Stages:
  1  base matrix (train_base.py, A/C/B x seeds, 30k steps each)
  2  qk lineage weights (gen_qk_lineage.py)
  3  native replay (replay_native.py)
  4  K selection (select_k_modes.py)
  5  five primary gates (K01/K02/K03/K07/K14 run.py)
  6  S1/S2 (M10/M11 run.py)
  7  verdict roll-up (ten_seed_verdict.json)

Every stage skips itself if its output marker already exists, so the
driver is resumable after interruption.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
PY = sys.executable


def sh(cmd, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    print(f'\n$ {cmd}', flush=True)
    r = subprocess.run(cmd, shell=True, env=e)
    if r.returncode != 0:
        raise SystemExit(f'stage failed: {cmd}')
    return r


def done(fp):
    return os.path.exists(fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repro-dir', required=True,
                    help='dir that contains (or will contain) '
                         'bridge_zp_{A,C,B}/')
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage', type=int, default=0,
                    help='run only this stage (default: all)')
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    args = ap.parse_args()
    seeds = args.seeds
    out = args.out
    os.makedirs(out, exist_ok=True)
    S = args.stage

    def want(n):
        return S in (0, n)

    # ---- Stage 1: base matrix ------------------------------------
    if want(1):
        marker = os.path.join(out, 'stage1_base.done')
        if not done(marker):
            for family in ('A', 'C', 'B'):
                bdir = os.path.join(args.repro_dir,
                                    f'bridge_zp_{family}')
                for seed in seeds.split(','):
                    # idempotent per (family, seed): summary.json exists?
                    import glob
                    if glob.glob(os.path.join(
                            bdir, f'seed{seed}_*_summary.json')):
                        print(f'[stage1] skip {family} seed{seed} '
                              '(summary exists)')
                        continue
                    sh(f'"{PY}" "{PKG}/base/train_base.py" '
                       f'--family {family} --seed {seed} '
                       f'--steps 30000 --eval_every 2000 '
                       f'--out "{bdir}"')
            open(marker, 'w').write('ok')

    # ---- Stage 2: qk lineage -------------------------------------
    if want(2):
        if not done(os.path.join(out, 'qk_lineage', 'summary.json')):
            sh(f'"{PY}" "{HERE}/gen_qk_lineage.py" '
               f'--repro-dir "{args.repro_dir}" --out "{out}" '
               f'--seeds {seeds}')

    # ---- Stage 3: replay ------------------------------------------
    if want(3):
        if not done(os.path.join(out, 'replay_summary.json')):
            sh(f'"{PY}" "{HERE}/replay_native.py" '
               f'--repro-dir "{args.repro_dir}" --qk-dir "{out}" '
               f'--out "{out}" --seeds {seeds}')

    # ---- Stage 4: K selection -------------------------------------
    if want(4):
        kdir = os.path.join(out, 'frozen_k')
        marker = os.path.join(out, 'stage4_k.done')
        if not done(marker):
            sh(f'"{PY}" "{HERE}/select_k_modes.py" '
               f'--repro-dir "{args.repro_dir}" '
               f'--replay-summary "{out}/replay_summary.json" '
               f'--ckpt-dir "{out}" --out "{out}" --seeds {seeds}')
            open(marker, 'w').write('ok')

    # ---- Stage 5: five gates ---------------------------------------
    # ARCHIVE SAFETY: each claim's run.py writes results.pkl into its
    # own directory -- which currently holds the ORIGINAL 3-seed
    # archive.  Before the first 10-seed run of a claim, move the
    # archive to results_3seed.pkl (never deleted; REPRO_PLAN §10.4
    # "old numbers never deleted").  Idempotence marker = the 10-seed
    # sidecar file written after each successful run.
    def archive_and_mark(claim):
        d = os.path.join(PKG, 'claims', claim)
        src = os.path.join(d, 'results.pkl')
        arc = os.path.join(d, 'results_3seed.pkl')
        if os.path.exists(src) and not os.path.exists(arc):
            os.replace(src, arc)
            print(f'[archive] {claim}: results.pkl -> results_3seed.pkl')
        return os.path.join(d, 'results_10seed.pkl')

    if want(5):
        for claim in ('K01_fate_lock', 'K02_qk_carrier',
                      'K03_identity_geometry', 'K07_maintenance',
                      'K14_dose_response'):
            mark = archive_and_mark(claim)
            if done(mark):
                print(f'[stage5] skip {claim} (10-seed results exist)')
                continue
            env = {'CLAIM_SEEDS': seeds, 'REPRO_DIR': args.repro_dir,
                   'TIER_B': '1'}
            sh(f'"{PY}" "{PKG}/claims/{claim}/run.py"', env=env)
            # rename the fresh 3-seed-named pkl to the 10-seed marker
            fresh = os.path.join(PKG, 'claims', claim, 'results.pkl')
            if os.path.exists(fresh):
                os.replace(fresh, mark)

    # ---- Stage 6: S1/S2 --------------------------------------------
    if want(6):
        for claim in ('M10_leverage_trace', 'M11_wr_factorial'):
            mark = archive_and_mark(claim)
            if done(mark):
                print(f'[stage6] skip {claim} (10-seed results exist)')
                continue
            sh(f'"{PY}" "{PKG}/claims/{claim}/run.py" '
               f'--ckpt-dir "{out}/r131c_ckpt" '
               # frozen-K JSONs live under {out}/frozen_k/ (Stage 4
               # output), not {out}/ directly
               f'--frozen-k-dir "{out}/frozen_k" --seeds {seeds}')
            fresh = os.path.join(PKG, 'claims', claim, 'results.pkl')
            if os.path.exists(fresh):
                os.replace(fresh, mark)

    # ---- Stage 7: verdict roll-up ----------------------------------
    if want(7):
        verdict = {}
        for claim in ('K01_fate_lock', 'K02_qk_carrier',
                      'K03_identity_geometry', 'K07_maintenance',
                      'K14_dose_response', 'M10_leverage_trace',
                      'M11_wr_factorial'):
            fp10 = os.path.join(PKG, 'claims', claim,
                                'results_10seed.pkl')
            fp9 = os.path.join(PKG, 'claims', claim,
                               'results_9seed.pkl')
            fp3 = os.path.join(PKG, 'claims', claim, 'results.pkl')
            fp3b = os.path.join(PKG, 'claims', claim,
                                'results_3seed.pkl')
            if done(fp10):
                verdict[claim] = '10-seed DONE'
            elif done(fp9):
                # seed9 censored at Stage 4 (no native crossing ->
                # frozen-K selection undefined); S1/S2 ran on 0-8
                verdict[claim] = ('9-seed DONE '
                                  '(seed9 stage-4 censored)')
            elif os.path.exists(fp3b):
                verdict[claim] = ('10-seed run incomplete '
                                  '(3seed archived)')
            elif os.path.exists(fp3):
                verdict[claim] = ('3-seed archive intact; '
                                  '10-seed not started')
            else:
                verdict[claim] = 'MISSING'
        with open(os.path.join(out, 'ten_seed_verdict.json'),
                  'w') as f:
            json.dump({'seeds': seeds, 'artifacts': verdict}, f,
                      indent=2)
        print(json.dumps(verdict, indent=2), flush=True)
        print('roll-up written; per-claim verdicts are in each '
              'results.pkl + run stdout', flush=True)


if __name__ == '__main__':
    main()
