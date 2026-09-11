# scripts/ — 驱动与回填（重训后置，先占位）

状态：TODO（DEFERRED）。规划：
- run_all_10seed.sh：串行总驱动（GPU 纪律：一次一卡、用户放行制）；
- fill_ledger.py：重跑 summary → 回填 CLAIM_LEDGER 数字列（mean ± SD over seeds，永不删除）。
