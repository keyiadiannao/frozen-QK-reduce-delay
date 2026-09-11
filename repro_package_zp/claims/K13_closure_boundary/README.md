# K13 — closure + restorativity assay boundary

- **Claim**：ADD 重建 structured-lookup slow regime（m1≈0.98–0.99、4k 无 crossing）；frozen QK 下 50 步脉冲无法通过 disruption gate ⇒ restorativity 组件 non-diagnostic（assay 边界，非 ADD 失败）
- **源脚本**：`repro/r106_closure_regime.py`
- **预注册**：账目 §6ef/§6eg（EXTENSION_SUMMARY.md；PREREG 内容见账目 §6ef/§6eg（提取待办，哈希锁定于 MANIFEST））
- **存量输出**：`repro/r106_closure_regime_results.pkl`
- **体量注记**：本目录 results.pkl 145 MB（codes 全轨迹字段），为包内最大文件；verify.py 需要它做逐位校验，不要手动精简
- **状态**：DONE（2026-09-02 入包 + verify 逐位 PASS：codes/va/strat/R_ctrl/fork 全 bit-exact；PARTIAL 判决复现 i/ii/iii 3/3 命中）
- **论文位置**：§6
