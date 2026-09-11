# K01 — fate lock at t=200

- **Claim**：t=200 时 slow fate 已写入非 QK 子系统：冻结 QK 200→4000 仍 va≈0.30 不 crossing；QK 冻于 init 对照 1375–1600 crossing
- **源脚本**：`repro/amp_necess2x2.py`
- **预注册**：账目 STATE_SPACE Assay A (R51)（EXTENSION_SUMMARY.md；PREREG 原文见账目对应 §（提取待办；MANIFEST 已锁哈希））
- **存量输出**：`repro/amp_necess2x2_results.pkl`
- **状态**：DONE（2026-09-02 入包 + verify 逐位 PASS：results.pkl 与存档 pkl 全字段 bit-exact，含 gnorm 探针）
- **论文位置**：§3
