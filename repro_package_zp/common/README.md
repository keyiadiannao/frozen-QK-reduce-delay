# common/ — 唯一公共库

状态：**M2 提取完成 + 对账全绿（2026-09-02）**。四模块已从 repro/ 逐字提取，
对账结果（`../verify_extract.py` + `../base/train_base.py --verify-summary`）：

| 检查 | 结果 |
|------|------|
| C1 zp 任务表 == (a+b)%113、split 3831/8938 | PASS |
| C2 task.make_split == gates.build_train == T.main 构造（seeds 0–2 逐位） | PASS |
| C3 fresh-init state_dict 与 T 逐位（seeds 0–2）；bridge A/C/B seeds0–2 t=200 前向与 T 逐位（max\|Δlogit\|=0）；n_params=227826 | PASS |
| C4 optimizer decay/no_decay 分组与 T 逐名字一致 | PASS |
| C5 sampler 18/18 draws 过 V1/V2/V3/V4/V6 且与 r92_gate_sampler 逐位相等 | PASS |
| C6 GPU 全训练重放 30000 步 vs 存档 summary：A seed0 ✓、A seed1 ✓、C seed0 ✓、B seed0 ✓ —— **history/first_cross/final/best 全部逐位相等**（4/4 PASS） | PASS |

唯一登记的已知分歧：存档 `final_od_acc` 非 nan（旧版 od 定义，zp 交换律下 OD
子集恒空，现行代码正确返回 nan）——legacy 字段，不作失败项。

## 模块清单（出处均已写入各文件头注释）

| 模块 | 内容 | 主要来源（repro/） |
|------|------|--------------------|
| task.py | zp 任务表 / set_group / make_split（unsorted id_train）/ 张量构造 | train_d57_abeq.py L32–74、main() L380–391 |
| model.py | D57Model（全 pos_mode/arch 分支逐字）+ make_optimizer + accuracy/od_accuracy | train_d57_abeq.py L77–305 |
| states.py | checkpoint I/O / block 移植 / fork 深拷贝 / bridge 路径解析 | train_d57_abeq.py L421–445、R94/R106 fork 机制 |
| gates.py | ZPIndexing + build_train/classify/sample_pi/verify_assignment（V1–V4/V6） | r92_gate_sampler.py L47–179 |
| readouts.py | ZPReadouts：val_probe / plain_forward_attn / own_codes(CLR) / retrieve_self / margin_self / batch_chain（RNG 构造逐字，bit-exact 迁移后 K06/K07 重验 PASS） | r103/r104 共享机制（L57–129、L291–299） |

## 提取要点（改动前必读）：
1. **tag 陷阱**：bridge tag 段 `fp20` = 前缀 `fp2` + 值 `0`（freeze_p2_steps=0，
   无冻结）；首次重放曾误读为"冻结 20 步"导致偏离，已修正并以逐位重放证实。
2. **冻结默认参数**：D57Model 默认 `n_elem=N_ELEM` 在类定义时绑定 114——
   这是全部存档 checkpoint 的参数形状来源（n_params=227826），禁止改晚绑定。
3. **opt.state_dict() 活引用**：fork 必须深拷贝（states.fork_state）。
4. **§6cz-prime**：batch↔pair 对应消费 unsorted id_train（task.make_split
   返回 unsorted；gates.build_train 同）。

