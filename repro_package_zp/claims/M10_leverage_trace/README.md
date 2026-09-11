# M10 — S1 leverage trace（secondary replication port）

- **Claim**：J_K^novel 在 plateau 窗已为正且向 crossing 窗上升（R136a/M10）
- **源脚本**：`repro/r136a_k_leverage.py (§6i Z/AA/AB)`
- **判据**：REPRO_PLAN §10.2a（冻结+smoke 修正：窗口中位数口径）
- **状态**：PORT（2026-09-05 为 10-seed secondary replication 移植；smoke 通过；**无 verify.py**——discovery-grade Tier A 不适用，10-seed 后如需 seal 再补）
- **运行**：`python run.py --ckpt-dir <OUT>/r131c_ckpt --frozen-k-dir <OUT> --seeds 0,...,9`（前置 = Stage 3/4 产物；见 OUTPUT_CONVENTION.md §1.2）
- **论文位置**：§6.5
- **注意**：本目录 results.pkl 会被运行覆盖（当前为 smoke 清理后空缺）；3-seed 原始数字在 repro/r136a_results.pkl / r136b_results.pkl
