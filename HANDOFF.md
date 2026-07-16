# LeakBench-Tab — 交接文档 (Handoff)

**Date:** 2026-07-16
**GitHub:** https://github.com/lzzzhh/LeakBench-Tab
**Branch:** main
**Status:** SP5–SP8 证据链已冻结；SP8 manifest、claim matrix、bootstrap 与入口文档一致。

---

## 项目是什么

LeakBench-Tab 是一个表格预测泄漏基准。核心框架是 **C/D/X 三轴**:
- **C (Construction validity):** 特征在预测时是否合法
- **D (Statistical Detectability):** 泄漏特征能否被统计方法定位
- **X (Model Exploitability):** 模型实际利用泄漏特征的程度

覆盖 **11 种泄漏机制** (M01–M11)、**20 个数据集**、**7 个模型**。

---

## 各阶段完成情况

### SP5 — Frozen Core (27,500 cells, 5 模型)
**Commit 参考:** 从 `2059308` 到 `63f00c6` 区域

- 使用 `structured_prior_replacement_v1` 冻结协议修正了 M04/M05/M08 (常数 0.5 prior)
- TabM 模型通过 **WSL2 Ubuntu + CUDA** 运行 (Windows 路径 bug 不能直接跑)
- M10 通过 `m10_amendment` 修正了 strict-view (保留合法注入字段)
- 最终 ledger: `artifacts/sp5/claim_ledger_v2.csv` — 27,500 cells, 5 models (lr/rf/lightgbm/catboost/tabm)
- **Claim 状态:** CL2↓ CL3→ CL4→ CL10→ (全部重新计算并通过 paper macros 同步)

### SP6 — Modern Model Expansion (38,500 cells, 7 模型)
**Commit 参考:** 从 `41a61ea` 到 `07d6f5a` 区域

- **ModernNCA:** 从官方 LAMDA-TALENT repo vendor 逐字节模型代码 (`third_party/modernnca/`) + 最薄 adapter
- **TabR:** 官方独立 micromamba 环境 (`tabr-official-sp6`, torch 1.13.1/cu117/faiss-gpu 1.7.2), 子进程 bridge, y_test 永远不传给子进程
- **TabPFNv2/TabICL:** DEFERRED
- 旧实现 (ModernNCA=sklearn NCA, TabR=简化 kNN) 已归档排除
- 7 模型 profile consistency: mean pairwise Spearman 0.891
- **M04/M05 negative harm:** 首次发现跨三个现代模型独立复现

### SP7 — Negative-Harm Mechanism Study (已关闭)
**Commit 参考:** `61790c0` 到 `9996f46`

- **CL13a:** PARTIALLY_CONFIRMED (TabM/ModernNCA/TabR 方向一致为负, TabR CI 跨零)
- **CL13:** UNCONFIRMED (因果机制未建立)
- **H2/H3/H4:** NOT SUPPORTED (预注册哨兵干预全部未通过)
- 现象确认: M04/M05 的轻微负 harm 不是 TabM 特异, 但产生机制未知
- **已正式关闭:** 不创建 CL13b, 不追加解释性实验

### SP8 — Governance ✅ COMPLETE_FROZEN
**证据入口:** `artifacts/sp8/governance_clean_manifest.json`

- 旧治理 runner (`run_meta_tier.py`) 永久 INTEGRITY_HOLD
- 旧 bundle runner (`run_governance_bundle.py`): UNDER_AUDIT — non-oracle 路径读取 leakage_mask, NOT_APPLICABLE 被保存为 SUCCESS
- **Clean runner** (`run_sp8_clean.py`): oracle-isolated, P0/P1/P2/P3 matched-cost, proper strict_distance_reduction, 55,000 rows / 5,500 keys / 0 failures / 0 duplicates
- 旧 77,000 行: NON_CLAIM_ELIGIBLE (provenance only)
- **Claims:** G1/G3/G4 SUPPORTED；G2 INCONCLUSIVE
- P3 vs P2 @20%: diff +0.051, 95% paired dataset-cluster bootstrap CI[+0.008,+0.087], P(diff>0)=98.94%
- 机制异质性: simple +0.109 (CI excludes 0)；structured -0.003 (CI crosses 0)；boundary +0.044 (CI excludes 0)
- 哈希绑定: runner `6089aaca`；CSV `6e3aa4c7`；bootstrap `51c9e877`；analysis script `daae7c39`
- P4/P5 NOT_APPLICABLE (no operational lifecycle/provenance metadata in frozen bundles)

---

## 仓库布局

```
LeakBench-Tab/
├── src/leakbench/          # 核心: mechanisms, governance, diagnostics, models
│   ├── mechanisms/
│   │   ├── __init__.py              # 基础 injector (LeakBenchInjector)
│   │   └── structured_prior_v1.py   # 修正后 M04/M05/M08 (常数 0.5 prior)
│   └── governance/
│       └── __init__.py              # 治理模块 (已修正, 但旧 runner 仍 INTEGRITY_HOLD)
├── experiments/leakbench/   # 实验入口
│   ├── run_structured_prior_v1_bundle.py  # SP4/SP5 冻结 bundle runner (Windows 不通)
│   ├── run_corrected_core.py              # CPU core runner
│   ├── run_corrected_tabm.py              # TabM runner (code_hash 漂移)
│   ├── run_meta_tier.py                   # 旧治理 runner (INTEGRITY_HOLD!)
│   └── run_m10_amendment.py              # M10 strict-view amendment runner
├── scripts/                 # 脚本
│   ├── assemble_claim_ledger_inputs_v2.py  # SP5 合并 exact-replacement
│   ├── build_claim_ledger_v2.py           # SP5 ledger 构建
│   ├── recompute_sp5_claims.py            # CL2/CL3/CL4/CL10 重算
│   ├── render_sp5_figures.py             # SP5 图表
│   ├── compute_sp4_detectability.py       # SP4 detectability 从 frozen bundles
│   ├── run_sp6_bundle_model.py            # SP6 模型无关 bundle runner
│   ├── run_sp6_tabr_bridge.py             # SP6 TabR 子进程 bridge
│   ├── run_sp7_intervention.py            # SP7 干预实验 runner
│   ├── run_governance_bundle.py           # SP8 治理 bundle runner
│   ├── export_sp7_intervention_bundles.py # SP7 干预 bundle 导出
│   ├── build_sp6_bundle_manifest.py       # SP6 bundle manifest
│   └── analyze_sp8_governance.py          # SP8 governance bootstrap/claims
├── configs/                 # 配置
│   ├── paper/                # 冻结协议 configs
│   └── sp6/                  # ModernNCA/TabR configs
├── third_party/modernnca/   # ModernNCA 官方源码 vendor (逐字节, LAMDA-TALENT @08301d6)
├── artifacts/               # 正式产物
│   ├── sp5/                  # claim_ledger_v2, claim matrix, per-claim 分析
│   ├── sp5_5/                # 论文同步: claim lock, traceability, 文本审计
│   ├── sp6/                  # 扩展 ledger, 7-model analysis, TabR env lock
│   ├── sp7/                  # 预注册, 观察复现, 诊断, sentinel 干预, final lock
│   └── sp8/                  # 治理审计, policy registry, sentinel, core confirmation, claims
├── results/                 # 原始结果 (部分 gitignored)
│   ├── structured_prior_replacement_v1/  # SP4 冻结 bundle + model cells
│   └── corrected_v2/                     # core CPU cells, TabM checkpoints, task bundles
├── protocols/               # 冻结协议
│   └── structured_prior_v1/  # inference_protocol, freeze_manifest, task plans
├── paper/edbt_eab/          # 当前 EDBT EA&B 论文与三表 paper-facing boundary
├── reports/                 # 各种审计和报告
└── archive/                 # 排除/作废证据 (code drift, interim M08)
```

---

## 关键经验教训

### WSL2 + CUDA 环境
- **位置:** `ssh 123@192.168.1.111` → Windows 主机, WSL2 Ubuntu inside
- **GPU:** NVIDIA RTX 4060 Laptop, driver 560.94, CUDA 12.6
- **Python venv:** `/root/lbenv/` (torch 2.5.1+cu121, tabm 0.0.3, sklearn/pandas/numpy)
- **TabR 官方环境:** `/root/micromamba/envs/tabr-official-sp6/` (torch 1.13.1, faiss-gpu 1.7.2, python 3.9.16)
- **TabR 官方 repo:** `/root/external/tabular-dl-tabr/` (commit 17baa90)
- **SP7 工作区:** `/root/sp7/`
- **持久化运行:** 用 Windows `schtasks` 启动 WSL 命令, 因为 WSL2 在最后一个 session 退出时会关闭发行版
- **路径:** 项目必须在 WSL ext4 (`/root/...`), 不能在 `/mnt/c/...`

### 关键 bug 和修复
- **Windows 路径 bug:** `run_structured_prior_v1_bundle.py` 的 `_relative()` 用反斜杠, 冻结 manifest 用正斜杠 → WSL2 解决
- **ModernNCA dtype bug:** vendored 代码 `d_out=1` 时 `unsqueeze(-1)` 不转换 float → 手动修复 `unsqueeze(-1).to(x.dtype)`
- **TabR 环境依赖:** `environment-simple.yaml` mkl 冲突 → 改用官方 `environment.yaml` (完整 lock)
- **TabM code drift:** 旧 checkpoint code_hash `99b17868` ≠ 当前 runner → 全部排除, 从零重跑
- **Natural adapter:** `Path.rglob(...,recursive=True)` Python 3.13 不再支持 → 改为 `rglob("*.csv")`
- **Interim M08:** entity-mean 临时实现被冻结协议取代 → 归档排除

---

## 后续建议 (尚未执行)

### Priority 1 — SP8-D Natural Governance
- Natural-task adapter 已修复 (测试通过)
- 需要用 bundle-only 治理 runner 在自然数据集上验证
- 至少 2 个可评估自然任务

### Priority 2 — SP8-E Operational Analysis
- 治理策略的运行时/资源开销
- Metadata coverage sensitivity
- Scalability analysis

### Priority 3 — TabPFNv2 / TabICL (SP6 延后)
- TabPFNv2: 之前全 0 失败, 需要重新审计
- TabICL: 无官方实现

### Priority 4 — 治理扩展到 catboost/tabm
- 当前只验证了 LR
- 全 5 模型治理需要 495k cells (>100k limit) — 需要压缩方案

---

## 不可触碰的冻结内容

以下内容 **绝对不能修改:**
- `artifacts/sp5/claim_ledger_v2.csv` (SP5 核心 ledger)
- `artifacts/sp5/claim_evidence_matrix_v2.*` (SP5 Claim Matrix)
- `artifacts/sp6/claim_ledger_v3_extended.csv` (SP6 扩展 ledger)
- `protocols/structured_prior_v1/` (冻结协议)
- `experiments/leakbench/run_structured_prior_v1_bundle.py` (字节冻结 runner)
- `experiments/leakbench/run_meta_tier.py` (INTEGRITY_HOLD — 不可执行)
- `results/structured_prior_replacement_v1/task_bundles/` (冻结 bundle)
- `third_party/modernnca/UPSTREAM.md` 中声明的官方 commit

---

## 当前测试状态

```
281 passed, 3 skipped, 0 failed
Python 3.13, macOS + WSL2 Ubuntu
Pre-existing failures: 0 (已全部修复)
```

---

## 快速开始命令

```bash
# 重建 SP5 ledger
python scripts/assemble_claim_ledger_inputs_v2.py
python scripts/compute_sp4_detectability.py
python scripts/build_claim_ledger_v2.py
python scripts/recompute_sp5_claims.py

# 检查 EDBT 三个 paper-facing CSV 与冻结 claim state 一致
python paper/edbt_eab/source_data/build_paper_assets.py --check

# 重生成并编译 EDBT 论文产物
python paper/edbt_eab/source_data/generate_paper_artifacts.py
tectonic -X compile paper/edbt_eab/main.tex --outdir paper/edbt_eab/output

# 禁止使用旧 run_governance_bundle.py 刷新论文结论；它仍处于 UNDER_AUDIT
# 当前材料、禁用资产与 release blocker 见：
# reports/edbt_eab/PROJECT_MATERIALS.md

# 全部测试
python -m pytest tests/ -q
```
