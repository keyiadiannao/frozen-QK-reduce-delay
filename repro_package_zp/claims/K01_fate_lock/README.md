# K01 — fate lock at t=200

- **Claim**：到 t=200，slow/fast 分支已经确定——冻结 QK（200→4000）仍 va≈0.30 不 crossing，而 QK 冻于 init 的对照在 1375–1925 越过（10 seeds）；其后 QK 可塑性既非 fast 分支所必需，也不足以解除 slow 分支。**本实验不定位载体**：载体是 joint Q–K configuration，由 K02 的交互移植确定。（早前措辞「slow fate 已写入非 QK 子系统」把 necessity 结果过度解读成载体位置，已作废——见 `ERRATA.md` §1d。）
- **源脚本**：`repro/amp_necess2x2.py`
- **预注册**：账目 STATE_SPACE Assay A (R51)（EXTENSION_SUMMARY.md；PREREG 原文见账目对应 §（提取待办；MANIFEST 已锁哈希））
- **存量输出**：`repro/amp_necess2x2_results.pkl`
- **状态**：DONE（2026-09-02 入包 + verify 逐位 PASS：results.pkl 与存档 pkl 全字段 bit-exact，含 gnorm 探针）
- **论文位置**：§3
