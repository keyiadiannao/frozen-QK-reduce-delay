# base/ — base 训练规范（重训后置；先落规范文档）

状态：**DEFERRED**（10-seed 重训后置，用户 2026-09-02 裁定）。本目录当前只放
规范，不放可执行训练入口。

## SNAPSHOT_SPEC（base run 必须留下的产物，提取自存量 checkpoint 惯例）

每 (seed, config) 一个目录，命名沿用 `bridge_zp_*` 惯例：
`seed{N}_abeq[_noeqemb]_a0.0_eps1e-05_f{qk,wv}200?_fwv0_fmlp0_fp20_zeros_wd_0011_*`

| 产物 | 用途 | 服务 |
|------|------|------|
| `full0000200.pt` | t=200 完整快照（model+opt） | K01/K02/K04/K05/K07 移植与解剖的宿主/供体 |
| `full0000700.pt` | t=700 快照 | K08 扰动实验的 fork 起点 |
| `firstcross.pt` | 首 crossing checkpoint | 复现包外读者核对 crossing 定义 |
| `final.pt` | 终局 | T95/终局读数 |
| `qk_trace_0_200.pt` | **0:200 QK 逐步参数流** | K05 formation accounting 的必需输入（存量无此产物，重训时新增） |
| `summary.json` | 曲线 + 门记录 + sha256 | 对账与密封 |

## 配置矩阵（重训放行后执行）

- A（natS，fqk0）/ C（F，fqk200）/ B（Z，noeqemb）× seeds 0–9，CAP 30000
  （与存档 bridge 一致；此前写 14000 有误——bridge summary final_step=30000），
  eval_every 2000，wd 0.0011，zeros PE，**fp2=0（tag 段 "fp20" = 前缀 fp2 +
  值 0，无 p2 冻结）**。
- **M2 对账已按此口径通过**：`train_base.py --verify-summary` 对 A seed0/seed1、
  C seed0、B seed0 全部 30000 步逐位 PASS（2026-09-02）。
- 吞吐实测：30000 步 ≈ 100 s/臂（RTX 5060，torch 2.11.0+cu128，mamba2 env）
  ⇒ 全 10-seed 矩阵 GPU 预算相应下调（见 REPRO_PLAN §三修订待办）。
