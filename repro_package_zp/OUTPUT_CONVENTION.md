# OUTPUT_CONVENTION.md — 复现包输出位置规范（审稿人契约）

本文件规定 `repro_package_zp/` 内每一个脚本的每一步输出落在哪里。
审稿人按本文件应能在 30 分钟内：跑通 smoke、定位任何数字的来源文件、
区分"包自带结果"与"自己跑出的结果"。

## 0. 目录分层（一条铁律）

| 层 | 位置 | 性质 | 修改权限 |
|----|------|------|---------|
| **包自带（sealed）** | 各 claim 目录内的 `results.pkl` / `verify.py` | 3-seed 存档结果 + 逐位校验器，与 manifests/MANIFEST.json 哈希对应 | **只读**。重跑会覆盖它——必须先读 §2 |
| **重跑输出（rerun）** | `results_3seed.pkl`（自动改名保底）+ `results_10seed.pkl` | 10-seed confirmatory 产物 | 驱动器自动管理 |
| **运行时产物** | `<OUT>` 目录（`--out` 参数指定） | bridge/replay/K 选择等大数据 | 随意，可删可重跑 |

**铁律：审稿人若只想核对包自带结果，永远不要在包目录内直接跑 `run.py`；
要么用 `verify.py`（bit-exact 校验，seal 机），要么用驱动器的
`--stage`（自动归档保底）。**

## 1. 每个脚本的输出对照表

### 1.1 五 primary gates（claims/K01, K02, K03, K07, K14）

| 脚本 | stdout | 文件输出 |
|------|--------|---------|
| `claims/K01_fate_lock/run.py` | 四臂 crossing 表 | `claims/K01_fate_lock/results.pkl`（**覆盖式**，见 §2） |
| `claims/K02_qk_carrier/run.py` | per-arm verdict + gate 行 | 同上（K02 目录） |
| `claims/K03_identity_geometry/run.py` | branch 分类 + G0''/G-S/G-F + `G-verdict tierB`（TIER_B=1 时） | 同上（K03 目录） |
| `claims/K07_maintenance/run.py` | 维持性判据行 | 同上（K07 目录） |
| `claims/K14_dose_response/run.py` | 剂量结构表 | 同上（K14 目录） |

环境变量：`CLAIM_SEEDS`（默认 `0,1,2`）、`REPRO_DIR`（bridge 锚点目录）、
`TIER_B=1`（服务器模式；K03 必需）。

### 1.2 Secondary（claims/M10, M11）

| 脚本 | 必需参数 | 文件输出 |
|------|---------|---------|
| `claims/M10_leverage_trace/run.py` | `--ckpt-dir <OUT>/r131c_ckpt --frozen-k-dir <OUT>` | `claims/M10_leverage_trace/results.pkl`（覆盖式） |
| `claims/M11_wr_factorial/run.py` | 同上 | `claims/M11_wr_factorial/results.pkl`（覆盖式） |

前置：`<OUT>/replay_summary.json`（每 seed 须含 `cross` 字段）与
`<OUT>/frozen_k/seed{N}.json`（Stage 3/4 产物）。

### 1.3 Stage 脚本（scripts/）

| 脚本 | 输出位置 | 说明 |
|------|---------|------|
| `base/train_base.py --family F --seed N --out DIR` | `DIR/bridge_zp_{A,C,B}/seedN_*.pt` + `*_summary.json`（含 sha256）+ `*_firstcross.pt` + `*_qk_trace_0_200.pt` | 每臂 ~40 MB |
| `scripts/gen_qk_lineage.py --repro-dir --out` | `<OUT>/qk_lineage/seedN_{F_QS,F_Q0}.pt` + `summary.json`（含 gates + sha256） | r107 权重物化 |
| `scripts/replay_native.py --repro-dir --qk-dir --out` | `<OUT>/r131c_ckpt/s{N}_t{T}.pt` + `<OUT>/replay_summary.json`（cross + 每 ckpt sha256） | 大头，~1.5 GB |
| `scripts/select_k_modes.py --replay-summary --ckpt-dir --out` | `<OUT>/frozen_k/seed{N}.json`（K + diag_share + top8） | |
| `scripts/run_ten_seed.py --repro-dir --out [--stage N]` | 全部上述 + 完成标记 `<OUT>/stage{N}_*.done` + `<OUT>/ten_seed_verdict.json` + 各 claim 的 `results_10seed.pkl` | 推荐入口 |

### 1.4 校验

| 脚本 | 功能 | 何时可用 |
|------|------|---------|
| `claims/*/verify.py` | 重跑 vs 存档 pkl 全字段逐位（Tier A） | 仅 seal 机（同 torch/GPU）；其它机器跑会报 MISMATCH，属预期（ENVIRONMENT.md 两层契约） |
| `scripts/seal.py <tag> <note>` | 全包 sha256 重封 | 修改包内任何文件后必须重封 |

## 2. 重跑 10-seed 的正确姿势（对审稿人）

```bash
# 不要手动跑单个 run.py（会覆盖 sealed results.pkl）！
# 正确方式：驱动器（自动把旧 results.pkl 改名为 results_3seed.pkl 保底）
python scripts/run_ten_seed.py --repro-dir /path/bridges --out /path/out
# 分 stage 同理：
python scripts/run_ten_seed.py --repro-dir ... --out ... --stage 5
```

产物对照：
- 旧 3-seed 结果 → `claims/<claim>/results_3seed.pkl`（永不删除）；
- 新 10-seed 结果 → `claims/<claim>/results_10seed.pkl`；
- roll-up → `<OUT>/ten_seed_verdict.json`（三态：10-seed DONE /
  run incomplete / 3-seed archive intact）。

## 3. 数字溯源（审稿人查一个数字的路径）

1. 论文里的每个 claim 有 `\claimk{Kxx}` / `Mxx` 标签；
2. → `paper_zp/CLAIM_LEDGER.md` 查该编号行：得到 源脚本 / 存档 pkl /
   账目 § / 论文节；
3. → 包内 `claims/<claim>/`：README（claim 声明）+ run.py（复现）+
   results.pkl（存档数）；
4. → 过程叙事在 `repro/EXTENSION_SUMMARY.md` 对应 §。

## 4. 包卫生规范

- `__pycache__/` 不入包（seal.py 已排除；分发前 `find . -name __pycache__ -exec rm -rf {} +`）；
- smoke/临时文件不得留在包内（`_smoke*`、`*.tmp` 零容忍）；
- `K12_surrogate_skip/` 无 run.py/verify.py 是**设计如此**（SKIP 状态，
  README 说明），不补；
- M10/M11 暂无 verify.py：它们是 3-seed discovery-grade（不入 confirmatory
  family），Tier A 逐位校验不适用；10-seed 后如需 seal 再补。

## 5. 分发清单（打包给审稿人时）

```
repro_package_zp/            # 全部（manifests/ 为密封凭证）
paper_zp/main.pdf + main.tex + CLAIM_MATRIX.md + CLAIM_LEDGER.md
paper_zp/EXPERIMENT_CENSUS.md + SECTION_AUDIT_MAP.md + REPRO_PLAN.md
paper_zp/TEN_SEED_RUN_PLAN.md + OUTPUT_CONVENTION.md（本文件）
repro/EXTENSION_SUMMARY.md   # 过程叙事账目
```
不分发：`repro/` 全部训练 checkpoint 与中间 pkl（~数十 GB；claims 所需
锚点哈希已记录在 MANIFEST，审稿人用 Tier B 行为判据即可复现）。

## 6. 2026-09-07 新增产物

| 产物 | 位置 | 说明 |
|------|------|------|
| 种子表型分析（S1 post hoc 注记的数据支撑） | `claims/M10_leverage_trace/seed_phenotype_analysis.json` + `seed_phenotype_spectral.json` | 结构轴（diag_share_L 双峰）/ 慢轴；论文 §mech-leverage "A post hoc reading" 一句的数字来源 |
| 表型再生脚本 | `claims/M10_leverage_trace/phenotype_table.py` | 从运行时 `tenseed_out/`（frozen_k/、replay_summary.json）+ 包内 M10/M11/K03 pkl 重算上表；`--r131c-ckpt` 可选做 s9 谱检查 |
| CB2 修正版结果 | `wd1111_battery/cb2_transplant/cb2_results.json` | Carrier-Holds 3/3（zero_grad 修复后重跑）；无效运行留档为同目录 `cb2_results_INVALID_zerograd_bug.json` |
