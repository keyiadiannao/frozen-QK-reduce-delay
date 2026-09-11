# K07 — frozen Q_S -> identity-maintaining dynamics

- **Claim**：冻结 Q_S carrier 下下游真实梯度动力学为 identity-maintaining。（**两套锚点，勿混用**）3 seeds（`results.pkl`，下方"存量输出"所指）：I(500) QS 0.965–0.970 vs Q0 0.658–0.686；10 seeds（`results_10seed.pkl`）：I(500) QS 0.934–0.981 vs Q0 0.470–0.934——论文与短文引用的是后者。
- **源脚本**：`repro/r103_frozen_carrier_retention.py`
- **预注册**：账目 §6dt/§6dx（EXTENSION_SUMMARY.md；PREREG 原文见账目对应 §（提取待办；MANIFEST 已锁哈希））
- **存量输出**：`repro/r103_frozen_carrier_retention_results.pkl`
- **状态**：DONE（2026-09-02 入包 + verify 逐位 PASS：results.pkl 与存档 pkl 全字段 bit-exact）
- **论文位置**：§5
