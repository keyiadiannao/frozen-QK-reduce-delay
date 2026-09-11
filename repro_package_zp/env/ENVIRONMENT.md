# ENVIRONMENT — 两层可复现性契约（v1.1，2026-09-02 用户裁定修订）

## 两层契约

| 层 | 内容 | 适用环境 | 判据 |
|----|------|----------|------|
| **Tier A**（bit-exact anchor） | 每个 claim 的 `verify.py`：重跑结果 vs 存档 pkl 全字段逐位 | **仅 seal 机**（同机/同 torch 2.11.0+cu128/同 GPU RTX 5060） | max\|Δ\|=0 |
| **Tier B**（verdict / statistical） | 重跑判据复现：crossing 落带、fate 六格分类、剂量结构、各预注册门通过、mean ± SD over seeds | **任何环境**（租用服务器、审稿人机器；CPU 自动回退） | 预注册行为判据（全部为行为级，天然跨硬件） |

Tier A 是 M2/M2.5 的**会计工具**（证明提取无漂移），不是包的运行要求；
Tier B 是 10-seed 重跑与对外发布所用的标准。**跨硬件逐位破是预期行为，
不构成失败**。

## 实测（seal 机）环境

| 项 | 值 |
|----|-----|
| Python | conda env `mamba2`（`C:/Users/26433/miniconda3/envs/mamba2/python.exe`） |
| numpy / torch | 2.4.3 / 2.11.0+cu128（CUDA 12.8 wheel） |
| GPU | NVIDIA RTX 5060 8GB（单卡串行纪律） |
| CPU 线程 | `torch.set_num_threads(4)`（脚本内固化） |

## 租用服务器跑 10-seed 的操作口径

1. 服务器不需要 `repro/` 存档区。包内 `base/train_base.py` 可从零生成
   base 训练（A/C/B 配置），claim 脚本通过 `REPRO_DIR` 环境变量或包内
   `expected/` 获取锚点；Tier B 模式下不读外部存档 pkl。
2. 逐位门（G-natS/G-S/G-F/prefix/G2'/G-C 等）在服务器上自动退化为
   **跨 seed 一致性 + 判据门**——门结构不变，bitwise 比较对象改为
   本次 server run 的 seed 间一致性（实现：claim 脚本已按"门通过=布尔"
   输出，服务器核对 verdict 行即可）。
3. 产出按 **mean ± SD over seeds，永不删除** 回填 ledger（fill_ledger
   流程，REPRO_PLAN §三）。

## 已知确定例外（任何 tier 都不作为失败项）

1. 存档 `final_od_acc`：旧版 od 定义（zp 交换律下 OD 子集恒空，现行正确
   返回 nan）——`train_base.verify_against_summary` 显式登记；
2. r98 bridge-anchor Z 为 **lineage-diff**（设计如此，§6dl 登记）。

## 吞吐实测（seal 机，预算校准用；服务器按 GPU 等级折算）

| 工作 | 实测 |
|------|------|
| base 训练 30000 步 | ≈100 s/臂 |
| K03 r85 全程 | ≈5 min/2 seeds |
| K08 r94 全程 | ≈14 min |
| K11 r98 全程 | ≈16.5 min |
| K09 r92+r93 | ≈22.5 min |
| K10 r84+r86 | ≈20 min |

## 运行纪律

- GPU 一次一卡、串行（seal 机）；服务器可按租用配额并行 arms；
- seal 后包冻结：只接受对账修复，新实验进 repro/ 不进包。
