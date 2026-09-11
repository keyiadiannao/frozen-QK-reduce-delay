# -*- coding: utf-8 -*-
"""Package seal script (reusable).  Usage:
    python scripts/seal.py <version-tag> <change-note>
Seals repro_package_zp: hashes every package file (excluding manifests/)
plus the external anchors (archived pkls + bridge checkpoints), writes
manifests/MANIFEST.json, CHECKSUMS.txt, SEAL.txt.  Run from anywhere.
"""
import glob
import hashlib
import json
import os
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'dev'
    note = sys.argv[2] if len(sys.argv) > 2 else ''
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    files = {}
    for r, dirs, fns in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in ('__pycache__',)]
        for fn in sorted(fns):
            p = os.path.join(r, fn)
            rel = os.path.relpath(p, '.').replace('\\', '/')
            if rel.startswith('manifests/'):
                continue
            files[rel] = sha256(p)

    ANCHORS = [
        'repro/amp_dirswap_results.pkl',
        'repro/amp_necess2x2_results.pkl',
        'repro/r80_swap_fixedpi_results.pkl',
        'repro/r84_persistence_missing_cell_results.pkl',
        'repro/r85_revisit_identity_results.pkl',
        'repro/r86_lifetime_dose_results.pkl',
        'repro/r92_native_ingraph_ws_results.pkl',
        'repro/r93_fixedws_release_results.pkl',
        'repro/r94_perturb_recovery_results.pkl',
        'repro/r97_state_factorial_results.pkl',
        'repro/r98_pretransition_assay_results.pkl',
        'repro/r102_dm_dissect_results.pkl',
        'repro/r103_frozen_carrier_retention_results.pkl',
        'repro/r104_native_addremove_results.pkl',
        'repro/r105_formation_accounting_results.pkl',
        'repro/r106_closure_regime_results.pkl',
        'repro/r107_code_amplitude_results.pkl',
        'repro/r108_plastic_qk_bridge_results.pkl',
        'repro/r108b_body_reacquisition_results.pkl',
        'repro/r108c_oper_amplitude_results.pkl',
        'repro/r109_gradient_field_results.pkl',
    ]
    for fam in ('A', 'C', 'B', 'D'):
        for pat in ('*.pt', '*_summary.json'):
            for p in sorted(glob.glob(f'../repro/bridge_zp_{fam}/{pat}')):
                norm = p.replace('\\', '/').lstrip('./')
                if norm.startswith('repro/') and norm not in ANCHORS:
                    ANCHORS.append(norm)

    anchors, missing = {}, []
    for rel in ANCHORS:
        ap = os.path.join('..', rel)
        if os.path.exists(ap):
            anchors[rel] = sha256(ap)
        else:
            missing.append(rel)

    manifest = {
        'seal': tag,
        'sealed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'package': 'repro_package_zp',
        'claim': 'Zp early-attention programming (TMLR submission), '
                 'claims K01-K16 per paper_zp/CLAIM_LEDGER.md',
        'recontract': 'two-tier reproducibility: Tier A bit-exact (seal '
                      'machine only), Tier B verdict/statistical '
                      '(portable, default for 10-seed servers)',
        'change': note,
        'claims_in_package': ['K%02d' % i for i in range(1, 17)],
        'n_files': len(files),
        'files_sha256': files,
        'external_anchors_sha256': anchors,
        'external_anchors_missing': missing,
    }
    os.makedirs('manifests', exist_ok=True)
    with open('manifests/MANIFEST.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    with open('manifests/CHECKSUMS.txt', 'w', encoding='utf-8') as f:
        for k in sorted(files):
            f.write(f"{files[k]}  {k}\n")
        for k in sorted(anchors):
            f.write(f"{anchors[k]}  {k}  (external anchor)\n")
    mh = sha256('manifests/MANIFEST.json')
    with open('manifests/SEAL.txt', 'w', encoding='utf-8') as f:
        f.write(f"seal: {tag}\n"
                f"sealed_at_utc: {manifest['sealed_at_utc']}\n"
                f"recontract: two-tier (Tier A bit-exact local / Tier B "
                f"verdict portable)\n"
                f"claims: K01-K16\n"
                f"change: {note}\n"
                f"n_files: {len(files)}\n"
                f"n_external_anchors: {len(anchors)}\n"
                f"external_anchors_missing: {len(missing)}\n"
                f"MANIFEST.json sha256: {mh}\n")
    print(f"sealed {tag}: {len(files)} files, {len(anchors)} anchors, "
          f"missing={len(missing)}")
    print(f"MANIFEST sha256 = {mh}")
    if missing:
        print("!! missing anchors -- seal NOT valid:")
        for m in missing:
            print('   ', m)
        sys.exit(1)


if __name__ == '__main__':
    main()
