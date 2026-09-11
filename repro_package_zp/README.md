# repro_package_zp/ — 复现包骨架（仅复现，不研究）

> **审稿人入口**：输出位置规范见 `OUTPUT_CONVENTION.md`（每个脚本的每一步输出落在哪里、如何避免覆盖 sealed 结果、数字溯源路径）。

日期：2026-09-02 立；2026-09-07 更新。状态：**K01–K16 + M10/M11 全部迁入并复现完毕**；
10-seed 确认、WD1111 battery、种子表型分析均已入包（见下）。定位与准入判据
见 `../paper_zp/REPRO_PLAN.md`（唯一权威）；本 README 只管包内结构与内容流。

## 定位三原则（用户裁定）

1. **仅用于复现，不用于研究**——新实验永远进 `repro/`，不进包；
2. **只有值得进论文的才进包**——准入以 `../paper_zp/CLAIM_LEDGER.md` 的 K 行为准；
3. **10-seed 重训后置**——先把包结构、提取对账、分类秩序做好。

## 包结构（目标态）

```
repro_package_zp/
├── README.md                 # 本文件
├── env/                      # requirements + 实测软硬件说明
├── common/                   # 唯一公共库：task / model / states / gates
├── base/                     # base 训练入口（重训后置，先只放规范文档）
├── claims/                   # 一条 K 一个目录（K01–K13；K12 SKIP）
├── manifests/                # MANIFEST.json（sha256）+ CHECKSUMS
└── scripts/                  # run_all 驱动 + fill_ledger 回填工具
```

## 未来内容流向表

| 对象 | 落位 | 规则 |
|------|------|------|
| `common/` 提取物 | common/{task,model,states,gates}.py | 从 repro/ 各脚本**去重提取**；每条带原出处（文件+行号）注释；提取后必须先用存量 seed0–2 与存档 pkl **逐位对账**，对不齐不得进包 |
| claim 复现脚本 | claims/Kxx/run.py | 从 repro/rXX_*.py 复制改造为包内调用 common/ 的版本；原脚本不动 |
| 预注册 | claims/Kxx/PREREG.md | 照抄账目 §原文（判据 + 分类规则 + 锁定哈希），禁止事后改写 |
| 对账/验证 | claims/Kxx/verify.py | 产出 summary.json + mean±SD + 门报告 |
| 存量数字 | claims/Kxx/expected/ | 存量 2–3 seed 对照带（只参照，不作门） |
| base 重训 | base/train_base.py | **后置**：10-seed 放行后才实现；先落 SNAPSHOT_SPEC（t=200/700 快照 + QK 逐步参数流） |
| 密封 | manifests/ | 全包 sha256；包冻结后只接受对账修复 |

## claims/ 目录现状（13 条，与 ledger 同号）

每目录含 README（claim 声明 + 源脚本指针 + 状态）。状态含义：
`TODO-EXTRACT`（待提取改造）/ `SKIP`（不进包，引用存量账目）/ `DONE`。

## 下一步（顺序锁定）

1. `common/` 四模块提取 + seed0–2 逐位对账（需用户放行）；
2. K01–K13 逐条改造 + 对账（每条完成即更新其 README 状态与 ledger rerun 列）；
3. env/ 固化（requirements + 吞吐实测）；
4. manifests/ 首 seal（此时包已可独立复现存量结论）；
5. **10-seed 重训与重跑**（后置，放行后按 REPRO_PLAN §三矩阵执行）。

## 首 seal（2026-09-02）

- **SEAL v1.1-first**：69 包文件 + 67 外部锚点全 sha256，0 缺失；MANIFEST sha256 = `c3411255987ecaf2f734d798a65da0de5ac4e0029b398a31657765feb352c778`（见 manifests/SEAL.txt）。
- **两层契约**（用户裁定）：Tier A bit-exact 仅 seal 机；Tier B verdict/statistical 可移植——10-seed 租用服务器默认跑 Tier B，逐位破是预期行为不构成失败。
- 包自 seal 起冻结。


---

# 10-seed 服务器运行（2026-09-05 整理新增节）

冻结协议：`paper_zp/REPRO_PLAN.md §十` + `paper_zp/TEN_SEED_RUN_PLAN.md`（执行层）。
契约：`env/ENVIRONMENT.md`（Tier B = 服务器标准；跨硬件逐位破是预期行为）。

## 一句话跑法

```bash
cd repro_package_zp
python scripts/run_ten_seed.py --repro-dir /path/bridges --out /path/tenseed_out
```

- `--repro-dir`：bridge 输出目录（Stage 1 自动生成 A/C/B × seeds 0–9；
  已有 summary.json 的臂自动跳过）。
- 断点续跑：每 stage 有完成 marker，重跑同命令自动跳过已完成部分。

## 分 stage 跑（推荐，便于监控）

```bash
python scripts/run_ten_seed.py --repro-dir ... --out ... --stage 1   # base ~50min
python scripts/run_ten_seed.py --repro-dir ... --out ... --stage 2   # qk lineage ~2min
python scripts/run_ten_seed.py --repro-dir ... --out ... --stage 3   # replay ~40min
python scripts/run_ten_seed.py --repro-dir ... --out ... --stage 4   # K selection ~1min
python scripts/run_ten_seed.py --repro-dir ... --out ... --stage 5   # 5 gates ~3.5h
python scripts/run_ten_seed.py --repro-dir ... --out ... --stage 6   # S1/S2 ~3-4h
python scripts/run_ten_seed.py --repro-dir ... --out ... --stage 7   # roll-up
```

## 环境要求

Python ≥ 3.10, numpy, torch（CUDA 可用则用；CPU 回退但慢）；
磁盘 ~3 GB。

## 判据（冻结，跑前跑后一字不改）

- **Primary（P1–P5）**：K01/K02/K03/K07/K14 各自 run.py 内置 gate
  （措辞见 CLAIM_MATRIX.md §A；TIER_B=1 下 K03 的 G-S/G-F 用 within-run
  一致性替代外部锚比较，判据语义不变）。
- **S1**：J_novel(P=t×−6000) > 0 且 J_novel(L=t×−1000) > J_novel(P)，
  per-seed 中位数；≥7/10 seeds 通过记 PASS。
- **S2**：I_match > 0 于 ≥7/10 seeds（raw；RMS-matched 只报告）。
- K 集选择：Stage 4，L 点 write 场对角能量 top-5（R134b canonical，
  signed-orbit 四格投影，unary 剔除）。
- 统计口径：mean ± SD over seeds，**永不删除** 3-seed 旧数字；
  checkpoint 数量永不进入方差估计。

## 本次整理新增/修改的文件

| 文件 | 性质 |
|------|------|
| `scripts/run_ten_seed.py` | 新增（驱动器） |
| `scripts/gen_qk_lineage.py` | 新增（Stage 2；r107 协议物化） |
| `scripts/replay_native.py` | 新增（Stage 3；r131c phase1 移植 + 强制 P/L 补存点） |
| `scripts/select_k_modes.py` | 新增（Stage 4；R134b diag_measures 移植） |
| `claims/M10_leverage_trace/run.py` | 新增（S1 = r136a 移植） |
| `claims/M11_wr_factorial/run.py` | 新增（S2 = r136b 移植） |
| `base/train_base.py` | 修改（SNAPSHOT_SPEC 兑现：firstcross.pt / qk_trace_0_200.pt / summary sha256） |
| `claims/K0{1,2,3}_*/run.py`、`K07/K14 run.py` | 修改（CLAIM_SEEDS 参数化；K03 加 TIER_B 分支） |

以上均为"对账修复/协议兑现"性质（seal 冻结例外条款），无判据语义改动。

## 服务器注意事项

1. K03 在 TIER_B=1 下只替换**外部锚比较对象**（G-S/G-F 改为 within-run
   跨 seed 一致性报告）；**K3 的 verdict 仍走预注册 fixed-lag 判据**
   （snap=200 释放后 ΔP@100 > 0 且 ΔP@500 > 0），可用
   `python repro/k03_offline_verdict.py --path <results_10seed.pkl>`
   离线复评。⚠️ 早期版本曾把 "natF 早 crossing + natS censored" 当作
   TIER_B 判据——那是自撰判据，会产出**伪 GATE FAILURE**，已作废；
   见 `paper_zp/TEN_SEED_RUN_PLAN.md §J-5`。
2. 任何 gate FAIL：只登记 + 复核器械，**不得改判据重跑**（§10.4）；
3. S1/S2 的 `replay_summary.json` 必须含每 seed 的 `cross` 字段
   （Stage 3 自动生成）；M10/M11 断言 checkpoint 落点在 offset ±200 步内，
   若断言失败说明 Stage 3 强制补存点缺失——查 replay 日志，勿放宽容差；
4. 跑完回传：各 `claims/*/results_10seed.pkl`（M10/M11 为
   `results_9seed.pkl`）+ `tenseed_out/replay_summary.json`
   + `ten_seed_verdict.json` + 全部 stdout 日志。
   **命名纪律**：3-seed 存档保留 `results_3seed.pkl`；sealed 结果只增不覆盖
   （`OUTPUT_CONVENTION.md`）。

## 10-seed 结果（2026-09-06，AutoDL RTX 4090-24G）

K01/K02/K07/K14 判据**通过**；K03 的身份门（G0''、tierB 前缀门、fixed-lag
10/10）通过，但**分支齐性门失败**（8 SHORT-RANGE-ONLY + 2 LATE，natS 3 seeds
censored 纯视界），如实记录于 ten_seed_verdict.json；
S2（I_match）PASS（9/9），S1（J_K 轨迹）**BELOW GATE 6/9** 如实降级
（失败种子属低结构表型，见 claims/M10_leverage_trace/seed_phenotype_analysis.json
与 phenotype_table.py）。逐 seed 数字与验收对账：`../paper_zp/TEN_SEED_RUN_PLAN.md §J`。

## WD1111 Core Battery (v1.10-k16-mfix-wb addition)

Scripts: scripts/train_wd1111_bridge.py + scripts/run_wd1111_battery.py
         + scripts/cb2_transplant.py + scripts/cb34_readouts.py

Purpose: verify that the paper's primary claims survive under
attention-inclusive weight decay (wd_1111).  Four experiments:
  CB1  fate contrast (natS vs natF, from scratch)
  CB2  carrier transplant (R97 2×2 on wd_1111-born t=200 snapshots)
  CB3  lookup M2y (pair-specific vs class-shared write code)
  CB4  Fourier ladder (L1 spectral concentration + L2 reconstruction
       + L3 diagonal factorization)

Run (after 10-seed completes, or in parallel if GPU allows):
  python scripts/run_wd1111_battery.py \
      --out /path/wd1111_battery --seeds 0,1,2

Stages: W1 (bridge training) → CB1 → CB2 → CB3 → CB4 → summary.
Output: {out}/wd1111_battery_summary.json + per-CB subdirectories.

Preregistration: EXTENSION_SUMMARY §6i AS (this document).

**Result (2026-09-06, 3 seeds, non-gate / exploratory)**——four sub-items
all ran to completion:
  CB1  fate contrast **reproduces** under wd_1111 (natF cross 4000 on
       3/3; natS 8000 / 10000 / 12000) — the slow/fast split is not an
       artifact of selective decay;
  CB2  **Carrier-Holds 3/3**（修正版重跑 2026-09-06，CAP2 4000→20000 预揭盲
       扩展，QK 源= t=200 快照，R97/R125 语义）：原生 t=200 配置装到哪个
       身体都慢跨（S 体 8400–15925，F 体 11050–13275），初始构造装到哪个
       身体都快跨（1150–1950，与 R125 seed-0 的 1425–1650 吻合）。
       事故记录：早前两次运行因 cb2_transplant.py 漏 opt.zero_grad() 无效
       （梯度跨步累积卡死在 chance），其产物保留为
       cb2_results_INVALID_zerograd_bug.json，结论以修正版为准；
  CB3  at the final (30k) checkpoint Γ is necessary (REMOVE collapses to
       0.12–0.27) while the surviving content is **class-shared**
       (CENTROID 0.999–1.00, DELTA-ONLY 0.13–0.34);
  CB4  ladder instrument runs (L1 n50 = 4/5/5; K8 reconstruction
       0.994–1.000; diag share 0.890–0.975).
Per-item registration: `../paper_zp/EXPERIMENT_CENSUS.md` §Ⅱ-b.
