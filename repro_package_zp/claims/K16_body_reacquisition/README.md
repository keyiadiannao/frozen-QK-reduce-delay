# K16 — released-family split: the body controls re-acquisition

- **Claim**：同一 harness、同一 $Q_0$、同一释放协议（batch 链 `SeedSequence([9000+seed])`、CAP 6000），**只换 body**：
  - $(N_S,Q_0,\text{released})$ → **slow**（cens ×3，R108/K15）；
  - $(N_F,Q_0,\text{released})$ → **fast**（cross **1475/1700**，1 seed 发散 cens，va 0.006 平坦——如实登记）。
  判据（跑前锁定）：**SPLIT** = F 臂 cross ≥2/3（S 臂已知 cens 3/3）⇒ **实测 SPLIT 2/3 命中**。
- **含义**：frozen 族里 fate 跟 QK 走（K2/R97：body 不携带 fate）；released 族里 **body 控制 carrier 能否被重建** ⇒ **carrier（冻结延续下携带 fate）与 reacquisition competence（释放后重建 carrier 的能力）是状态的两种不同性质**。自然的两分：早期 QK 学习写下 (i) QK functional 本身 + (ii) 与之共适应的下游状态；后者可能帮助 (i) 在释放后重新形成（强提示，未隔离）。
- **源脚本**：`repro/r108b_body_reacquisition.py`（§6ej，2026-09-03；复用 R108 harness，核心逻辑逐字；包内版把 base 模块别名替换为直连 `common/`——对象与数值逐一相同）
- **预注册**：账目 §6ej（SPLIT/NOSPLIT 判据跑前锁定）
- **存量输出**：`repro/r108b_body_reacquisition_results.pkl`
- **状态**：DONE（2026-09-03 入包 + 全量重跑数值一致 + verify **32 checks / 0 failures / PASS**，含 λ_eff 轨迹与 A_oper 诊断；第七轮补的 λ_eff 诊断：F body 场沿学习方向完全不生长，全程 max|λ_eff| ≤ 0.056，S 臂 max 0.77–0.93——分离在场层级确立）
- **论文位置**：§5.6（`sec:program-release`，Table `tab:release` 第 5 行）
- **取向无关诊断（§6ek，第七/九轮）**：`oper_amplitude.py`——λ_eff 是 S 方向投影，取向无关的 A_oper(t)（role-mean unary score 幅度）显示 **两个 body 都再生 operand 场**（S max 205–306 逼近自然尺度 277–282；F max 50–80 ≈ 自然 1/4），fate 仍按 body 分离 ⇒ 场幅度不是 plastic 系统的控制变量，共适应状态才是；"code 回归"措辞降级为 "re-entry into the slow trajectory"。va/cross 与存档逐位一致（一致性锚）。

## 一致性门

`model_F` 内建断言：F@200 的 W_Q/W_K 与 init 的 max|Δ| < 1e-8（实测 0.0）——fqk200 冻结前 200 步无更新，故 QK 矩天然为 0，无混合态、无清零决策。

## 复跑

```bash
python run.py            # ~55 s on RTX 5060 (3 seeds x 6000 steps)
python verify.py         # vs repro/r108b_body_reacquisition_results.pkl
```
