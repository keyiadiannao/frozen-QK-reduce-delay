# K15 — plastic release: does the installed amplitude survive release?

- **Claim**：QK 释放后，安装幅度在**轨迹**上留下分级印记（λ_eff 的 max/mean 对 λ 单调，3/3 seeds），但**端点表型不可分**——每一臂（含 λ=0）在 6000 步内都进入 structured-lookup plateau（m1≈1.00、m2≈0.001–0.04、va≈0.30、全部 cens）。**最强的一条**：frozen L000（K14，cross 1875–2050）↔ released L000（本实验，6000 步内不 cross，λ_eff 从 0 长到 mean 0.28–0.41）——同一 t=200 构造、同一 batch 链前缀，唯一差别是 QK 是否继续可塑 ⇒ **释放后 block 自己重建 operand code，delay 随之回来**。
- **源脚本**：`repro/r108_plastic_qk_bridge.py`（§6ei，2026-09-03；核心逻辑逐字复制，仅 import shim / REPRO_DIR / SMOKE 变量 / 输出路径不同）
- **预注册**：账目 §6ei（PERSIST / WASHOUT / MIXED 判据跑前锁定；实测 **MIXED**——端点判据不成立，如实登记）
- **存量输出**：`repro/r108_plastic_qk_bridge_results.pkl` + `r108_plastic_qk_bridge.log`
- **状态**：DONE（2026-09-03 入包 + 全量重跑 8m09s 数值一致 + verify **366 checks / 0 failures / PASS**）
- **论文位置**：§5.6（`sec:program-release`）

## 设计要点（为什么不是"把 9 臂阶梯解除冻结再跑一遍"）

冻结 K14 的优点是 QK 参数干预与 optimizer 动力学解耦；实测确认 t=200 checkpoint **含原生 Adam 状态**（18 params），释放后参数=λ 编辑值、矩=原生 S 值，混合态是默认行为。所以：

- **(P) `rst` 主梯**：body 原生 Adam 历史**保留**、W_Q/W_K 矩**清零** → 全身无混合态；
- **(N) `nat` 真值参考**：原生状态原封不动（native continuation）；
- **(F) `fr` 稳健性**：全新 AdamW、零历史。

主读数 = **score-functional 幅度投影** λ_eff(t)（可塑后 h_2 也在动，v=W_K^T q_2 不是充分对象）；构造上 t=200 恰等于 λ（自检 0.000/0.500/1.000）。v-空间投影同报（兼容 K14 口径）。**G-RESET 门**记录前 50 步 QK 更新范数（矩清零的首步 ≈0.43·lr，跨臂完全相同，不构成 λ 混淆）。horizon 6000（不碰 Δ3/escape）。

## 复跑

```bash
python run.py            # ~8 min on RTX 5060 (3 seeds x 8 arms x 6000 steps)
python verify.py         # 366 checks vs repro/r108_plastic_qk_bridge_results.pkl
```

环境见 `env/ENVIRONMENT.md`（两层契约：Tier A bit-exact 仅 seal 机；Tier B verdict 可移植）。
