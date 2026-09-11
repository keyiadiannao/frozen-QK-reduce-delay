# M11 — S2 write-reader factorial（secondary replication port）

- **Claim**：matched late write-reader 系统 I_match>0（R136b/M11）
- **源脚本**：`repro/r136b_kw_read_factorial.py (§6i AC/AD)`
- **判据**：REPRO_PLAN §10.2a（I_match>0 于 ≥7/10 seeds）
- **状态**：PORT（2026-09-05 为 10-seed secondary replication 移植；smoke 通过；**无 verify.py**——discovery-grade Tier A 不适用，10-seed 后如需 seal 再补）
- **运行**：`python run.py --ckpt-dir <OUT>/r131c_ckpt --frozen-k-dir <OUT> --seeds 0,...,9`（前置 = Stage 3/4 产物；见 OUTPUT_CONVENTION.md §1.2）
- **论文位置**：§6.5
- **注意**：本目录 results.pkl 会被运行覆盖（当前为 smoke 清理后空缺）；3-seed 原始数字在 repro/r136a_results.pkl / r136b_results.pkl
