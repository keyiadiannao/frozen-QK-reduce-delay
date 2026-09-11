# manifests/ — 密封（v1.8-k16，2026-09-03）

- **SEAL v1.8-k16**：89 包文件 + 72 外部锚点全 sha256，0 缺失；MANIFEST sha256 = `ecdf8ad5…edb1a`。
- 相对 v1.7：新增 **R110 粗粒度 body 定位**（无双向定位；V/D 单向破 slow；R_S 反向加速；分布式 + 易破难授 ⇒ Δ1 定位线关闭）。
- 重封：`python scripts/seal.py <tag> <note>`。两层契约不变。包自 seal 起冻结。
- **SEAL v1.14-tenseed**（2026-09-06）：119 包文件 + 99 外部锚点全 sha256，0 缺失；MANIFEST sha256 = `04159896…46b24`。相对 v1.13：10-seed confirmatory 产物入包（K01/K02/K03/K07/K14 → results_10seed.pkl；M10/M11 → results_9seed.pkl；tenseed_out/ 与 wd1111_battery/）；K03 TIER_B 判据修正回预注册 fixed-lag；λ=0.5 删失数勘误（6/10）。
