"""scripts/run_all.py -- one-command pipeline for the full 10-seed
confirmatory (wd_0011) + WD1111 Core Battery.

Usage (server, inside tmux/screen):
    nohup python scripts/run_all.py \
        --repro-dir /path/to/bridges \
        --out /path/to/results \
    > pipeline.log 2>&1 &

Resumable: every stage has a completion marker; re-running skips
completed stages.  If the process dies, just re-run the same command.

Total estimated time: ~4 hours on a modern GPU (A100/4090).
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
PY = sys.executable


def sh(cmd):
    print(f'\n{"="*60}\n$ {cmd}\n{"="*60}', flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True)
    dt = time.time() - t0
    status = 'OK' if r.returncode == 0 else f'FAIL({r.returncode})'
    print(f'\n[{status}] {dt:.0f}s  {cmd[:80]}', flush=True)
    if r.returncode != 0:
        raise SystemExit(f'Stage failed: {cmd}')


def main():
    ap = argparse.ArgumentParser(
        description='Full 10-seed + WD1111 pipeline')
    ap.add_argument('--repro-dir', required=True,
                    help='dir containing bridge_zp_{A,C,B}/ (wd_0011)')
    ap.add_argument('--out', required=True,
                    help='output root for all results')
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--skip-wd1111', action='store_true',
                    help='skip the WD1111 battery (wd_0011 only)')
    args = ap.parse_args()
    seeds = args.seeds
    out = args.out
    os.makedirs(out, exist_ok=True)

    log = os.path.join(out, 'pipeline.log')
    t_start = time.time()
    print(f'[pipeline] start at {time.strftime("%Y-%m-%d %H:%M:%S")}',
          flush=True)
    print(f'[pipeline] seeds={seeds} out={out}', flush=True)
    print(f'[pipeline] log={log}', flush=True)

    # ---- Phase I: wd_0011 10-seed confirmatory ----
    print('\n' + '=' * 60)
    print('PHASE I: wd_0011 10-seed confirmatory')
    print('=' * 60, flush=True)
    sh(f'"{PY}" "{HERE}/run_ten_seed.py" '
       f'--repro-dir "{args.repro_dir}" '
       f'--out "{out}" --seeds {seeds}')

    # ---- Phase II: WD1111 Core Battery ----
    if not args.skip_wd1111:
        print('\n' + '=' * 60)
        print('PHASE II: WD1111 Core Battery')
        print('=' * 60, flush=True)
        sh(f'"{PY}" "{HERE}/run_wd1111_battery.py" '
           f'--out "{os.path.join(out, "wd1111")}" '
           f'--seeds {seeds} --device {args.device}')

    # ---- summary ----
    dt_total = time.time() - t_start
    h, m = divmod(int(dt_total), 3600)
    summary = {
        'status': 'COMPLETE',
        'duration_hms': f'{h}h{m:02d}m',
        'seeds': seeds,
        'out': out,
        'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    fp = os.path.join(out, 'pipeline_summary.json')
    with open(fp, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n{"="*60}')
    print(f'PIPELINE COMPLETE in {h}h{m:02d}m')
    print(f'Results: {out}')
    print(f'Summary: {fp}')
    print(f'{"="*60}', flush=True)


if __name__ == '__main__':
    main()
