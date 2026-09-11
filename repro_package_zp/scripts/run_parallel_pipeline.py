"""scripts/run_parallel_pipeline.py — 6-way parallel 10-seed pipeline
for RTX 4090 24GB server.

Usage (on server, inside tmux/screen):
    nohup python scripts/run_parallel_pipeline.py \
        --repro-dir /root/autodl-tmp/bridge_zp \
        --out /root/autodl-tmp/tenseed_out \
        --workers 6 \
        > pipeline.log 2>&1 &

Design:
  - Stage 1 (base training): 30 runs (3 families × 10 seeds), run
    6-way parallel via subprocess pool.  Each worker sets
    torch.set_num_threads(2) to avoid CPU oversubscription.
  - Stage 2-4: sequential (fast setup steps, <2 min total).
  - Stage 5 (five gates): K01/K02/K07/K14 are GPU-bound and
    parallelizable (4+4+2+9 arms = independent subprocesses);
    K03 is offline-heavy and run separately.
  - Stage 6 (M10/M11): offline checkpoint analysis, sequential.
  - Stage 7: verdict roll-up.

Threading:
  OMP_NUM_THREADS=2 per worker (env var); 6 workers × 2 = 12 CPU
  threads total.  GPU is shared (tiny model, ~500MB per context).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)


def gpu_env(worker_id):
    """Environment for a parallel worker: limit CPU threads."""
    e = dict(os.environ)
    e['OMP_NUM_THREADS'] = '2'
    e['MKL_NUM_THREADS'] = '2'
    e['WORKER_ID'] = str(worker_id)
    return e


def run_cmd(cmd, env_extra=None, log_fp=None):
    """Run a command, optionally teeing output to a log file."""
    e = dict(os.environ)
    e['OMP_NUM_THREADS'] = '2'
    e['MKL_NUM_THREADS'] = '2'
    if env_extra:
        e.update(env_extra)
    t0 = time.time()
    if log_fp:
        with open(log_fp, 'a') as f:
            f.write(f'\n$ {cmd}\n')
            f.flush()
        r = subprocess.run(cmd, shell=True, env=e,
                           stdout=open(log_fp, 'a'),
                           stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(cmd, shell=True, env=e)
    dt = time.time() - t0
    status = 'OK' if r.returncode == 0 else f'FAIL({r.returncode})'
    print(f'  [{status}] {dt:.0f}s  {os.path.basename(cmd.split()[-1])}',
          flush=True)
    return r.returncode == 0


def parallel_run(cmds, max_workers=6, log_dir=None):
    """Run a list of shell commands in parallel with a bounded pool."""
    if not cmds:
        print('  (nothing to run)', flush=True)
        return True
    print(f'  running {len(cmds)} tasks, {max_workers} workers...',
          flush=True)
    t0 = time.time()
    all_ok = True
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for i, cmd in enumerate(cmds):
            log_fp = os.path.join(log_dir, f'worker_{i:03d}.log') \
                if log_dir else None
            fut = pool.submit(run_cmd, cmd, gpu_env(i), log_fp)
            futures[fut] = cmd
        for fut in as_completed(futures):
            ok = fut.result()
            if not ok:
                all_ok = False
                print(f'  FAILED: {futures[fut][:80]}', flush=True)
    print(f'  done in {time.time() - t0:.0f}s '
          f'({"ALL OK" if all_ok else "SOME FAILED"})', flush=True)
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repro-dir', required=True,
                    help='dir containing bridge_zp_{A,C,B}/')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--skip-wd1111', action='store_true')
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    out = args.out
    log_dir = os.path.join(out, 'logs')
    os.makedirs(out, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    PY = sys.executable
    seed_str = args.seeds
    t_start = time.time()

    def log(msg):
        dt = time.time() - t_start
        h, m = divmod(int(dt), 3600)
        print(f'\n[{h:02d}:{m:02d}] {msg}', flush=True)

    # ================================================================
    # STAGE 1: base training (30 runs, 6-way parallel)
    # ================================================================
    log('STAGE 1: base training (3 families × 10 seeds)')
    cmds = []
    for family in ('A', 'C', 'B'):
        bdir = os.path.join(args.repro_dir, f'bridge_zp_{family}')
        os.makedirs(bdir, exist_ok=True)
        for seed in seeds:
            existing = glob.glob(os.path.join(
                bdir, f'seed{seed}_*_summary.json'))
            if existing:
                print(f'  skip {family} seed{seed} (exists)')
                continue
            cmds.append(
                f'"{PY}" "{PKG}/base/train_base.py" '
                f'--family {family} --seed {seed} '
                f'--steps 30000 --eval_every 2000 '
                f'--out "{bdir}"')
    if cmds:
        if not parallel_run(cmds, args.workers, log_dir):
            raise SystemExit('Stage 1 had failures')
    else:
        print('  all base runs already exist')
    open(os.path.join(out, 'stage1.done'), 'w').write('ok')

    # ================================================================
    # STAGE 2: QK lineage (sequential, fast)
    # ================================================================
    log('STAGE 2: QK lineage weights')
    if not done(os.path.join(out, 'qk_lineage', 'summary.json')):
        if not run_cmd(f'"{PY}" "{HERE}/gen_qk_lineage.py" '
                       f'--repro-dir "{args.repro_dir}" '
                       f'--out "{out}" --seeds {seed_str}',
                       log_fp=os.path.join(log_dir, 'stage2.log')):
            raise SystemExit('Stage 2 failed')

    # ================================================================
    # STAGE 3: native replay (sequential per seed, 10 total)
    # ================================================================
    log('STAGE 3: native F_QS replay with checkpoint saving')
    if not done(os.path.join(out, 'replay_summary.json')):
        if not run_cmd(f'"{PY}" "{HERE}/replay_native.py" '
                       f'--repro-dir "{args.repro_dir}" '
                       f'--qk-dir "{out}" --out "{out}" '
                       f'--seeds {seed_str}',
                       log_fp=os.path.join(log_dir, 'stage3.log')):
            raise SystemExit('Stage 3 failed')

    # ================================================================
    # STAGE 4: K mode selection (offline, fast)
    # ================================================================
    log('STAGE 4: K mode selection')
    kdir = os.path.join(out, 'frozen_k')
    if not done(os.path.join(out, 'stage4_k.done')):
        if not run_cmd(f'"{PY}" "{HERE}/select_k_modes.py" '
                       f'--repro-dir "{args.repro_dir}" '
                       f'--replay-summary "{out}/replay_summary.json" '
                       f'--ckpt-dir "{out}" --out "{out}" '
                       f'--seeds {seed_str}',
                       log_fp=os.path.join(log_dir, 'stage4.log')):
            raise SystemExit('Stage 4 failed')
    open(os.path.join(out, 'stage4_k.done'), 'w').write('ok')

    # ================================================================
    # STAGE 5: five primary gates
    # ================================================================
    log('STAGE 5: five primary gates')

    # Archive 3-seed results before 10-seed overwrites
    for claim in ('K01_fate_lock', 'K02_qk_carrier',
                  'K03_identity_geometry', 'K07_maintenance',
                  'K14_dose_response'):
        d = os.path.join(PKG, 'claims', claim)
        src = os.path.join(d, 'results.pkl')
        arc = os.path.join(d, 'results_3seed.pkl')
        if os.path.exists(src) and not os.path.exists(arc):
            os.replace(src, arc)
            print(f'  [archive] {claim}: results.pkl → results_3seed.pkl')

    # K01/K02/K07/K14 are GPU-bound and parallelizable
    gate_cmds = []
    gate_names = []
    for claim in ('K01_fate_lock', 'K02_qk_carrier',
                  'K07_maintenance', 'K14_dose_response'):
        log_fp = os.path.join(log_dir, f'{claim}.log')
        gate_cmds.append(
            f'"{PY}" "{PKG}/claims/{claim}/run.py"')
        gate_names.append(claim)

    # run K01/K02/K07 in parallel, then K14 (heaviest) alone
    env_gate = {'CLAIM_SEEDS': seed_str, 'REPRO_DIR': args.repro_dir,
                'TIER_B': '1'}
    log('  running K01/K02/K07 in parallel...')
    parallel_cmds = [f'"{PY}" "{PKG}/claims/K01_fate_lock/run.py"',
                     f'"{PY}" "{PKG}/claims/K02_qk_carrier/run.py"',
                     f'"{PY}" "{PKG}/claims/K07_maintenance/run.py"']
    if not parallel_run(parallel_cmds, 3, log_dir):
        print('  WARNING: some parallel gates had issues', flush=True)

    log('  running K14 (dose-response, heaviest)...')
    if not run_cmd(f'"{PY}" "{PKG}/claims/K14_dose_response/run.py"',
                   env_extra=env_gate,
                   log_fp=os.path.join(log_dir, 'K14.log')):
        print('  WARNING: K14 had issues', flush=True)

    # K03 (identity retrieval) — offline-heavy
    log('  running K03 (identity retrieval)...')
    if not run_cmd(f'"{PY}" "{PKG}/claims/K03_identity_geometry/run.py"',
                   env_extra=env_gate,
                   log_fp=os.path.join(log_dir, 'K03.log')):
        print('  WARNING: K03 had issues', flush=True)

    # archive 10-seed results
    for claim in ('K01_fate_lock', 'K02_qk_carrier',
                  'K03_identity_geometry', 'K07_maintenance',
                  'K14_dose_response'):
        d = os.path.join(PKG, 'claims', claim)
        src = os.path.join(d, 'results.pkl')
        mark = os.path.join(d, 'results_10seed.pkl')
        if os.path.exists(src) and not os.path.exists(mark):
            os.replace(src, mark)
            print(f'  [10seed] {claim}: results.pkl → results_10seed.pkl')

    # ================================================================
    # STAGE 6: M10/M11 (offline checkpoint analysis)
    # ================================================================
    log('STAGE 6: M10 (leverage trace) + M11 (write-reader factorial)')
    for claim in ('M10_leverage_trace', 'M11_wr_factorial'):
        d = os.path.join(PKG, 'claims', claim)
        src = os.path.join(d, 'results.pkl')
        arc = os.path.join(d, 'results_3seed.pkl')
        if os.path.exists(src) and not os.path.exists(arc):
            os.replace(src, arc)
        mark = os.path.join(d, 'results_10seed.pkl')
        if done(mark):
            print(f'  skip {claim} (10-seed exists)')
            continue
        if not run_cmd(f'"{PY}" "{d}/run.py" '
                       f'--ckpt-dir "{out}/r131c_ckpt" '
                       f'--frozen-k-dir "{out}" --seeds {seed_str}',
                       log_fp=os.path.join(log_dir, f'{claim}.log')):
            print(f'  WARNING: {claim} had issues', flush=True)

    # ================================================================
    # STAGE 7: verdict roll-up
    # ================================================================
    log('STAGE 7: verdict roll-up')
    verdict = {}
    all_claims = ('K01_fate_lock', 'K02_qk_carrier',
                  'K03_identity_geometry', 'K07_maintenance',
                  'K14_dose_response', 'M10_leverage_trace',
                  'M11_wr_factorial')
    for claim in all_claims:
        d = os.path.join(PKG, 'claims', claim)
        fp10 = os.path.join(d, 'results_10seed.pkl')
        fp3b = os.path.join(d, 'results_3seed.pkl')
        fp3 = os.path.join(d, 'results.pkl')
        if os.path.exists(fp10):
            verdict[claim] = '10-seed DONE'
        elif os.path.exists(fp3b):
            verdict[claim] = '10-seed incomplete (3seed archived)'
        elif os.path.exists(fp3):
            verdict[claim] = '3-seed archive intact; not started'
        else:
            verdict[claim] = 'MISSING'
    with open(os.path.join(out, 'ten_seed_verdict.json'), 'w') as f:
        json.dump({'seeds': seed_str, 'artifacts': verdict}, f,
                  indent=2)
    print(json.dumps(verdict, indent=2), flush=True)

    # ================================================================
    # PHASE II: WD1111 Core Battery (10 seeds)
    # ================================================================
    if not args.skip_wd1111:
        log('PHASE II: WD1111 Core Battery (10 seeds)')
        wd_out = os.path.join(out, 'wd1111')
        os.makedirs(wd_out, exist_ok=True)

        # W1: bridge training (20 runs, 6-way parallel)
        log('  W1: wd_1111 bridge training (A/C × 10 seeds)')
        w1_cmds = []
        for family in ('A', 'C'):
            bdir = os.path.join(wd_out, f'bridge_wd1111_{family}')
            os.makedirs(bdir, exist_ok=True)
            for seed in seeds:
                existing = glob.glob(os.path.join(
                    bdir, f'seed{seed}_*_summary.json'))
                if existing:
                    print(f'  skip wd1111 {family} seed{seed}')
                    continue
                w1_cmds.append(
                    f'"{PY}" "{HERE}/train_wd1111_bridge.py" '
                    f'--family {family} --seed {seed} '
                    f'--steps 30000 --out "{bdir}" '
                    f'--device {args.device}')
        if w1_cmds:
            if not parallel_run(w1_cmds, args.workers, log_dir):
                print('  WARNING: some W1 runs failed', flush=True)

        # CB1: fate contrast (JSON readout from W1)
        log('  CB1: fate contrast')
        cb1_dir = os.path.join(wd_out, 'cb1_fate')
        os.makedirs(cb1_dir, exist_ok=True)
        cb1 = {}
        for seed in seeds:
            for fam, key in (('A', 'natS'), ('C', 'natF')):
                fps = glob.glob(os.path.join(
                    wd_out, f'bridge_wd1111_{fam}',
                    f'seed{seed}_*_summary.json'))
                if fps:
                    s = json.load(open(fps[0]))
                    cb1.setdefault(seed, {})[key] = s.get('first_cross')
        with open(os.path.join(cb1_dir, 'cb1_results.json'), 'w') as f:
            json.dump(cb1, f, indent=2)

        # CB2: carrier transplant
        log('  CB2: carrier transplant')
        cb2_dir = os.path.join(wd_out, 'cb2_transplant')
        os.makedirs(cb2_dir, exist_ok=True)
        run_cmd(f'"{PY}" "{HERE}/cb2_transplant.py" '
                f'--bridge-dir "{wd_out}" --out "{cb2_dir}" '
                f'--seeds {seed_str} --device {args.device}',
                log_fp=os.path.join(log_dir, 'cb2.log'))

        # CB3/CB4: readouts on mature wd_1111 checkpoint
        log('  CB3: lookup M2y')
        cb3_dir = os.path.join(wd_out, 'cb3_lookup')
        os.makedirs(cb3_dir, exist_ok=True)
        run_cmd(f'"{PY}" "{HERE}/cb34_readouts.py" '
                f'--bridge-dir "{wd_out}" --out "{cb3_dir}" '
                f'--seeds {seed_str} --mode cb3 --device {args.device}',
                log_fp=os.path.join(log_dir, 'cb3.log'))

        log('  CB4: Fourier ladder')
        cb4_dir = os.path.join(wd_out, 'cb4_fourier')
        os.makedirs(cb4_dir, exist_ok=True)
        run_cmd(f'"{PY}" "{HERE}/cb34_readouts.py" '
                f'--bridge-dir "{wd_out}" --out "{cb4_dir}" '
                f'--seeds {seed_str} --mode cb4 --device {args.device}',
                log_fp=os.path.join(log_dir, 'cb4.log'))

    # ---- final summary ----
    dt_total = time.time() - t_start
    h, m = divmod(int(dt_total), 3600)
    summary_fp = os.path.join(out, 'pipeline_summary.json')
    with open(summary_fp, 'w') as f:
        json.dump({'status': 'COMPLETE',
                   'duration': f'{h}h{m:02d}m',
                   'seeds': seed_str,
                   'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
                   'verdicts': verdict}, f, indent=2)
    log(f'PIPELINE COMPLETE in {h}h{m:02d}m')
    log(f'Results: {out}')
    log(f'Summary: {summary_fp}')


def done(fp):
    return os.path.exists(fp)


if __name__ == '__main__':
    main()
