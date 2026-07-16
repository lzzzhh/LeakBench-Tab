# LeakBench-Tab: EDBT 2027 EA&B 实验与结论分析任务书

> 本文件可直接作为实验 Agent 的完整执行指令。  
> 工作目录：`/Users/zhanhuilin/Downloads/RiskAgent`  
> 目标投稿：EDBT 2027 Experiments, Analysis & Benchmarks（EA&B）  
> 目标轮次：第三轮，2026-10-07 17:00 Pacific Time  
> 论文写作负责人：Codex（当前主任务）  
> 实验、统计和结论分析负责人：接收本指令的 Agent

---

## 1. 你的角色与最终目标

你负责把 LeakBench-Tab 的实验、统计分析和 machine-readable evidence
完善到足以支撑 EDBT 2027 EA&B 投稿的程度。你不负责论文正文写作。

你的最终交付不是“更多实验数量”，而是一套满足以下条件的证据包：

1. benchmark 的数据生成、prediction-time validity gold standard 和污染标注可审计；
2. 核心模型、诊断方法、强度、机制和 seed 覆盖完整且 provenance 可追踪；
3. 统计单位、配对关系、置信区间和多重比较处理正确；
4. 所有结论都有唯一 machine-readable source，且明确适用范围；
5. negative/null/inconclusive 结果被诚实保留；
6. artifact 可由审稿人获取、运行和验证；
7. 论文作者可以仅依赖你交付的 claim/evidence package 写作，无需从日志或
   Markdown 中手抄数字。

EDBT EA&B 的贡献重点是对现有方法进行全面、严谨的实验评估，并给出其
strengths/weaknesses 的新洞见；benchmark 论文还应说明 benchmark 数据与 gold
standard 如何生成、如何使用，并提供代表性实验结果。不要把项目包装成一个新的
detector、tabular model 或 governance algorithm。

官方依据：

- CFP：<https://edbticdt2027.github.io/?contents=EDBT_CFP.html>
- Reviewer Guidelines：<https://edbticdt2027.github.io/?contents=EDBT_Reviewer_Guidelines.html>
- Important Dates：<https://edbticdt2027.github.io/?contents=important_dates.html>

---

## 2. 不可越过的角色边界

### 2.1 你可以做

- 检查、修复和运行实验代码；
- 恢复此前已经运行完成的真实结果；
- 新增必要的测试、validator、统计脚本和 artifact builder；
- 生成 CSV/Parquet/JSON/manifest/figure source data；
- 编写实验审计报告、claim matrix 和给论文作者的 evidence handoff；
- 在预注册后运行真正必要的补充实验；
- 根据证据将 claim 标成 `SUPPORTED`、`NOT_SUPPORTED`、`INCONCLUSIVE`、
  `DESCRIPTIVE_ONLY`、`CASE_STUDY_ONLY` 或 `BLOCKED`。

### 2.2 你不可以做

- 不写或改论文摘要、Introduction、Related Work、Conclusion 和主文叙事；
- 不修改当前 `paper/aaai27/` 下的 LaTeX 正文；该目录属于旧 AAAI 工作区；
- 不替论文作者决定最终章节结构或为了故事顺滑而删除负结果；
- 不手工编辑 paper-facing 数字、LaTeX macros 或最终 claim JSON；
- 不以“测试通过”替代科学有效性检查；
- 不因为旧 claim 已冻结就忽略新发现的证据冲突；发现冲突必须 fail closed；
- 未经用户明确要求，不执行 `git commit`、`git push`、创建分支或 PR。

你可以输出“允许进入正文的证据陈述”，但必须是短、受限、带 source path 的
claim record，而不是论文 prose。

---

## 3. 当前仓库基线（执行前必须重新核验）

以下是 2026-07-16 的已观测状态，不得直接当作最终事实；你必须从文件重新验证。

### 3.1 当前已存在

| Evidence block | 当前观测 |
|---|---|
| CPU confirmatory core | `results/corrected_v2/core_cpu_cells.csv`，22,000 行，4 models，0 duplicate keys |
| Official TabM core | `results/corrected_v2/tabm_bundle_confirmatory/tabm_cells.csv`，5,500 行，SHA prefix `45c05efc` |
| Diagnostic canonical matrix | `results/corrected_v2/diagnostic_canonical_cells.csv`，22,000 行，4 diagnostics |
| CPU M10 amendment | `results/corrected_v2/m10_amendment_confirmatory/cpu_cells.csv`，2,000 行 |
| Public natural evidence | `results/corrected_v2/public_natural/natural_cells.csv`，60 行 |
| SP8 governance | `artifacts/sp8/governance_clean.csv`，55,000 行 / 5,500 keys |
| SP8 claims | G1/G3/G4 `SUPPORTED`；G2 `INCONCLUSIVE` |
| Tests | 当前主任务观测为 280 passed，1 skipped，0 failed |

Official TabM 主表的当前 manifest 声明 5,500/5,500 SUCCESS，模型身份为
`tabm.TabM`，result SHA 为
`45c05efc573cc5055201bf366b22d2b9a1df0afb78b110ed58e015051b3c3351`。
不要因此跳过 manifest、code hash、environment 和 prediction 文件核验。

### 3.2 当前明确缺失或阻断

| 缺口 | 影响 |
|---|---|
| `results/corrected_v2/m10_amendment_confirmatory/tabm_cells.csv` | 缺少 M10 official TabM replacement 500 cells |
| 对应 `tabm_cells_manifest.json` | 无法验证 M10 TabM provenance |
| `results/corrected_v2/canonical_cells.csv` | 无法形成最终 27,500-cell core matrix |
| `results/corrected_v2/canonical_manifest.json` | 无 canonical hash chain |
| `results/corrected_v2/paper_claims.json` | paper-facing claim release fail closed |
| `results/corrected_v2/claim_state.json` | claim state 与 paper claim 无法 byte-identical 验证 |

此外，以下内容仍属于科学范围缺口，而不只是缺文件：

- SP8 governance 当前只验证 LR panel，不能外推到所有模型；
- lifecycle/provenance metadata 策略 P4/P5 当前 `NOT_APPLICABLE`，G2 必须保持
  `INCONCLUSIVE`，除非有新的、真正 operational metadata；
- natural-task governance 尚未完成；
- benchmark 的 runtime、memory、规模扩展和完整 reproduction cost 还没有形成
  EDBT-facing evidence block；
- public artifact 必须重新检查路径脱敏、license、数据来源和一键验证；
- 旧 77,000-row governance、旧 Meta、proxy models、pilot 和 superseded outputs
  均不能作为 claim evidence。

---

## 4. 工作树保护规则

当前工作树包含主任务尚未提交的 SP8.1 修复。开始前运行：

```bash
git status --short --branch
git diff --check
```

禁止 `git reset --hard`、`git checkout -- <path>`、`git clean`，也不得覆盖以下文件：

```text
README.md
HANDOFF.md
CURRENT_STATUS.md
scripts/analyze_sp8_governance.py
tests/test_sp8_governance.py
artifacts/sp8/bootstrap_analysis.json
artifacts/sp8/governance_clean_manifest.json
artifacts/sp8/sp8_audit_status.json
paper/aaai27/PAPER_WRITING_READINESS.md
```

这些文件可读取。若你认为其中存在新错误，记录到
`reports/edbt_eab/blockers.md`，不要直接覆盖。

所有新增 EDBT 实验、统计和审计输出优先放在：

```text
results/edbt_eab/
reports/edbt_eab/
artifacts/edbt_eab/
```

若既有 canonical builder 强制写入 `results/corrected_v2/`，可以使用其正式路径，
但必须先验证不会覆盖已有正式文件，并保留全部 source hashes。

---

## 5. 总体执行原则

1. **恢复优先于重跑。** 先搜索本地、Windows `E:\tabm`、WSL `/root` 和历史
   corrected-v2 输出。找到成品后先校验，不能仅按文件名接受。
2. **冻结优先于运行。** 任何新实验必须先写 protocol/config、预期矩阵、主指标、
   统计单位、停止条件和 hash，再观察结果。
3. **不允许 silent fallback。** 缺少模型依赖、GPU、真实数据或 metadata 时必须失败，
   不能退化成 synthetic/proxy/constant predictions 后保存为 SUCCESS。
4. **oracle 隔离。** 非 oracle 方法不得读取 leakage mask、污染位置、`n_clean`、机制
   标签、测试标签、未来标签或带有污染身份的特征名。
5. **matched comparison。** 对治理、诊断和模型比较保持相同 task、seed、budget、
   split 和可用信息。
6. **task 是主要推断单位。** 不得把 cell、seed 或 mechanism row 当作独立数据集。
7. **结论可以变弱。** 新结果不支持旧 claim 时应降级，禁止调参保结论。
8. **旧证据保留但隔离。** 不删除 legacy 文件；通过 deny-list、namespace、manifest
   和文档将其排除。

---

## 6. Workstream A：恢复并完成 canonical corrected-v2 release（P0）

### A1. Provenance inventory

先生成：

```text
reports/edbt_eab/provenance_inventory.md
reports/edbt_eab/provenance_inventory.json
```

必须列出：

- 每个缺失/已恢复产物的本地与远端候选路径；
- 生成命令、Python/PyTorch/CUDA/TabM 版本；
- runner/config/task bundle/environment SHA-256；
- 文件行数、schema、result SHA-256；
- 是“恢复既有产物”还是“重新运行”；
- 远端与本地复制前后的 hash 是否一致。

### A2. 恢复或运行 M10 official TabM amendment

目标输出：

```text
results/corrected_v2/m10_amendment_confirmatory/tabm_cells.csv
results/corrected_v2/m10_amendment_confirmatory/tabm_cells_manifest.json
```

验收条件：

- 精确 500 行：20 datasets × 1 mechanism × 5 strengths × 1 TabM × 5 seeds；
- 全部 `SUCCESS`，无 duplicate scientific keys/run IDs、NaN 或 constant placeholder；
- 模型身份为 official `tabm.TabM`；
- 使用冻结 M10 amendment protocol，不得使用原始错误 M10；
- strict/full policy、bundle hash、runner SHA、adapter SHA、config SHA、environment
  lock 全部匹配；
- `result_sha256` 与文件实际 SHA 一致；
- 与 original TabM M10 的 scientific keys 精确一一替换；
- 若需要 prediction files，数量、命名与 hash manifest 完整。

### A3. 构建 canonical 27,500-cell matrix

仅在 A2 通过后运行正式 builder：

```bash
python scripts/build_canonical_corrected_v2.py
```

目标输出：

```text
results/corrected_v2/canonical_cells.csv
results/corrected_v2/canonical_manifest.json
```

验收条件：

- 精确 27,500 行；
- 20 tasks × 11 mechanisms × 5 strengths × 5 models × 5 seeds；
- model set 精确为 `lr/rf/catboost/lightgbm/tabm`；
- 所有 scientific keys 唯一；
- 27,500 行全部 SUCCESS；
- M10 精确 2,500 行，且全部来自 amendment；
- CPU、TabM、M10 CPU、M10 TabM 与 task manifest 的 source hash 全部绑定；
- canonical manifest 的 `canonical_sha256` 与实际文件一致；
- builder 重新运行到另一个临时路径时输出应 byte-identical，或明确解释仅 manifest
  时间戳造成的差异。

### A4. 重建统计与 claim release

按仓库冻结协议重新运行 canonical-facing analyses。不得继续使用 hash 已过期的统计
输出。至少包括：

- category contrasts；
- mechanism/model summaries；
- diagnostic sensitivity；
- joint D--X analysis；
- M08/M09 cluster sensitivity；
- natural case-study summaries；
- multiplicity adjustment；
- provenance and integrity blocks。

之后运行：

```bash
python scripts/build_corrected_v2_claim_state.py --force
python scripts/validate_corrected_v2_release.py
```

目标：

```text
results/corrected_v2/paper_claims.json
results/corrected_v2/claim_state.json
```

二者必须 byte-identical，不能人工编辑。Validator 必须完整通过，不能使用
`--skip-tests` 作为最终验收。

---

## 7. Workstream B：Benchmark validity 与 gold-standard 审计（P0）

EA&B benchmark 不能只证明“代码能跑”，必须证明 benchmark 测量的是声明的对象。

生成：

```text
reports/edbt_eab/benchmark_validity_audit.md
artifacts/edbt_eab/mechanism_contract_matrix.csv
artifacts/edbt_eab/gold_standard_audit.json
```

对 M01--M11 逐机制记录：

- prediction-time boundary；
- 哪些字段无效以及为什么；
- 数据生成公式和 strength mapping；
- gold leakage mask 如何产生；
- strict view/full view 的字段差异；
- 是否依赖当前标签、未来标签、entity/source history；
- train/validation/test 划分和时间/实体同步规则；
- 每个 strength 的单调性预期与实际检查；
- deterministic seed contract；
- 现有单元测试和缺失测试；
- 允许的 paper-facing interpretation。

重点重新核验 M04/M05/M08/M09/M10：

- M04 不得将当前标签通过居中窗口回流；
- M05 必须与时间方向和 split 对齐；
- M08 的 entity assignment/rates 必须形成声明的实体结构，且跨 seed/model/strength
  同步规则与统计分析一致；
- M09 的 source categories 是 designed categories，不得当作随机 source population；
- M10 legitimate component、leak mask 和 strict-view amendment 必须一致。

若任何机制 contract 失败：

1. 立即将所有受影响 claim 标为 `BLOCKED`；
2. 写出影响矩阵；
3. 不继续 paper claim release；
4. 只修复根因并重跑受影响矩阵，不做无关重构。

---

## 8. Workstream C：EA&B baseline coverage 审计（P0）

生成：

```text
reports/edbt_eab/baseline_coverage_audit.md
artifacts/edbt_eab/baseline_matrix.csv
```

分别审计：

### C1. Diagnostic baselines

当前已有：

- mutual information；
- absolute correlation；
- LR coefficient magnitude；
- RF permutation importance。

检查这些方法是否覆盖：线性/非线性、model-free/model-based、cheap/expensive、
global/localized 等基本类别。基于公开文献和可复现实现，判断是否还缺少一个对
EDBT EA&B 结论有决定性影响的标准 baseline。

不要为了数量新增方法。只有同时满足以下条件才新增：

1. 是审稿人明显会期待的现有方法；
2. 有稳定公开实现或可审计的简单定义；
3. 能在当前 frozen tasks 上公平运行；
4. 预先冻结指标、超参数和失败处理；
5. 新增后确实改变 benchmark 对 strengths/weaknesses 的覆盖。

### C2. Model baselines

核心 paper-facing model set 当前固定为 LR、RF、CatBoost、LightGBM、official
TabM。ModernNCA/TabR 可以作为外部有效性扩展，但不能混入 core count；
TabPFNv2/TabICL 在没有真实成功运行前保持 deferred。

检查：

- official identity 和环境锁；
- preprocessing 是否只 fit train；
- early stopping 是否只用 validation；
- test labels 是否完全隔离；
- failed/non-finite/constant predictions 是否正确失败；
- 模型超参数是否固定且来源清楚。

### C3. Governance baselines

当前 claim-grade 策略为：keep、oracle、matched-cost random、blind MI。旧 group、
graph、lifecycle 结果不可用。

不要把 P4/P5 的 `NOT_APPLICABLE` 伪装成零效果。若没有真实 operational metadata，
G2 保持 `INCONCLUSIVE`。

---

## 9. Workstream D：统计与结论审计（P0）

生成：

```text
artifacts/edbt_eab/claim_evidence_matrix.json
artifacts/edbt_eab/claim_evidence_matrix.csv
reports/edbt_eab/statistical_audit.md
reports/edbt_eab/conclusion_handoff.md
```

### D1. 推断规则

- 主要 cluster unit 为 dataset/task；
- seed 是同一 task 内重复，不是独立 dataset；
- paired comparisons 必须复用相同 task/seed/draw；
- bootstrap probability 必须来自同一个 paired bootstrap distribution；
- `P(diff > 0)` 不是 frequentist p-value；
- category confirmatory tests 必须使用预注册规则和 multiplicity correction；
- mechanism-level、model-level、M08/M09 category reweighting 和 D--X 关系若未预注册，
  应标 `DESCRIPTIVE_ONLY`；
- designed registry 不能外推为未指定的任务/机制总体；
- null interval crossing zero 应写“no reliable advantage”，不能写“equivalent”或
  “does not work”；
- natural tasks 只允许 `CASE_STUDY_ONLY`，除非有真正抽样设计。

### D2. 每个 claim record 的必需字段

```text
claim_id
status
estimand
comparison
population_or_registry
unit_of_analysis
point_estimate
interval
interval_method
multiplicity_method
sample_size_by_level
source_paths
source_sha256
allowed_wording
prohibited_wording
limitations
main_or_supplement_recommendation
```

`main_or_supplement_recommendation` 只是给论文作者的建议，最终是否进入正文由论文
作者决定。

### D3. 必须回答的科学问题

1. C、D、X 是否确实提供不可互相替代的信息？
2. 哪些 leakage mechanisms 对哪些 diagnostics 构成稳定边界？
3. 诊断结论对 ranker choice 有多敏感？
4. exploitability 对 model family 有多敏感？
5. strength-response 是否一致，是否存在非单调或 sign reversal？
6. 简单/结构化差异是否在预注册推断下成立？
7. D--X 关系在控制 category、LOMO 和 joint uncertainty 后还能支持什么有限结论？
8. blind MI governance 相对 matched random 的优势是否跨 budget、mechanism category
   和 dataset 稳健？
9. LR-only governance 对 paper scope 的限制有多大？
10. natural case studies 与 controlled panel 是一致、冲突还是信息不足？

---

## 10. Workstream E：EDBT-facing 额外完善项（P1，必须先预注册）

以下是当前最可能被 EA&B 审稿人追问的缺口。先做 gap analysis，再决定是否运行；
不要直接大规模开跑。

### E1. Efficiency、scalability 与 reproduction cost

这是当前明显缺口。至少设计并冻结一个可复现的 profiling protocol，记录：

- task generation time；
- diagnostic runtime；
- model training/evaluation runtime；
- governance scoring/removal runtime；
- peak memory（可可靠测量时）；
- artifact size；
- 按 rows、features、leak fields 增长的规模曲线；
- CPU/GPU hardware 和软件环境；
- 完整 benchmark 与最小 smoke reproduction 的预计成本。

输出：

```text
results/edbt_eab/efficiency/
artifacts/edbt_eab/efficiency_manifest.json
reports/edbt_eab/efficiency_analysis.md
```

不要用不同机器的 wall time 直接做方法优劣比较；跨机器数据只能作为 reproduction
cost 描述。

### E2. Governance 跨模型稳健性

SP8 当前只覆盖 LR。先做 power/cost/design audit，提出一个最小、预注册、能回答
“LR 结果是否 model-specific”的方案，例如固定 20% budget 的受限 cross-model
confirmation，而不是盲目复制全部 55,000 行到所有模型。

必须提前冻结：模型集合、budget、mechanisms、primary contrast、最大 cell 数、统计
方法和成功标准。只有该实验能显著改善论文结论边界时才运行。

不得在看到结果后选择最有利模型或 budget。

### E3. Natural-task boundary checks

当前 public natural evidence 只有 60 行，应审计：

- 数据集名称、版本、公开来源、license 和 checksum；
- prediction-time boundary 和 contamination labels；
- 是否来自真实文件而非 synthetic fallback；
- preprocessing 与 categorical vocabulary 是否 train-only；
- 任务选择标准是否事先固定；
- 失败任务是否完整披露。

Natural-task governance 只有在至少两个合法、可复现、污染字段可评估的任务存在时才
运行。否则保持 deferred，不得用 weak anecdote 填表。

### E4. Robustness and sensitivity

检查是否需要以下最小敏感性分析：

- diagnostic seed / discretization / preprocessing；
- budget mapping 和 realized budget；
- metric choice（AUROC inflation、strict-distance、recall/retention）；
- dataset-level outlier / leave-one-task-out；
- mechanism category assignment；
- failed-run handling。

只补能影响结论的分析。不得把所有超参数网格都变成 paper contribution。

---

## 11. Workstream F：强制 artifact 与复现包（P0）

EDBT EA&B 要求提供 artifact。Artifact 不是最终才压缩的代码目录，而是实验验收的一
部分。

执行并完善：

```bash
python scripts/build_corrected_v2_artifact.py --zip
python scripts/verify_corrected_v2_public_artifact.py <artifact-path> --run-tests
```

最终 artifact 必须：

- 不包含用户名、私有绝对路径、IP、凭据、API key、访问日志；
- 包含 code、configs、freeze protocols、manifests、必要数据/生成器、claim state、
  tests 和最小 reproduction instructions；
- 对无法公开的数据给出公开获取方式、license 和 checksum，不得打包无授权数据；
- 有 quick-start smoke 路径和 full reproduction 路径；
- 说明 CPU/GPU 需求、预计时间、存储和内存；
- artifact 内部 hash manifest 全匹配；
- 从解压后的独立目录运行 verifier，而不是引用原仓库文件；
- public artifact 的 paper claims 与本地 claim state byte-identical；
- legacy/superseded evidence 不得被默认脚本读取。

输出：

```text
artifacts/edbt_eab/release_manifest.json
reports/edbt_eab/artifact_audit.md
```

---

## 12. 测试与质量门

最终至少运行：

```bash
python -m pytest tests/ -q -rs
python scripts/validate_corrected_v2_release.py
python scripts/verify_corrected_v2_public_artifact.py <artifact-path> --run-tests
git diff --check
```

还必须有针对以下契约的真实测试：

- mechanism gold mask 与 strict/full view；
- train/test/validation isolation；
- official model identity；
- complete scientific key coverage；
- M10 exact replacement；
- paired bootstrap probability；
- manifest hash binding；
- artifact path redaction；
- no silent fallback；
- deterministic regeneration。

不得用 `assert True`、空 `pass`、只检查文件存在或只检查行数的测试充当科学验证。

---

## 13. Claim 与状态管理规则

任何结论状态变更必须同时更新 machine-readable claim matrix 和 impact matrix。

### 允许状态

- `SUPPORTED`：预定义 decision rule 通过；
- `NOT_SUPPORTED`：预定义 directional claim 未通过；
- `INCONCLUSIVE`：设计或证据不足以支持二元判断；
- `DESCRIPTIVE_ONLY`：可报告 effect/interval，但不能做 confirmatory 推断；
- `CASE_STUDY_ONLY`：仅限具体自然任务边界；
- `BLOCKED`：存在 integrity/provenance/completeness 冲突；
- `NON_CLAIM_ELIGIBLE`：仅保留 provenance。

### 禁止状态操作

- 不因 CI 跨零而声称等效；
- 不因 point estimate 为负而声称反效果，除非 interval 和预注册规则支持；
- 不把 exploratory analysis 升为 confirmatory；
- 不把 oracle upper bound 写成 deployable method；
- 不把 LR-only 结论写成 model-general；
- 不把 fixed registry 写成 population inference；
- 不把 test pass 数写成 scientific validity 证据。

---

## 14. 最终交付目录

任务结束时必须至少交付：

```text
results/corrected_v2/m10_amendment_confirmatory/tabm_cells.csv
results/corrected_v2/m10_amendment_confirmatory/tabm_cells_manifest.json
results/corrected_v2/canonical_cells.csv
results/corrected_v2/canonical_manifest.json
results/corrected_v2/paper_claims.json
results/corrected_v2/claim_state.json

reports/edbt_eab/provenance_inventory.md
reports/edbt_eab/benchmark_validity_audit.md
reports/edbt_eab/baseline_coverage_audit.md
reports/edbt_eab/statistical_audit.md
reports/edbt_eab/conclusion_handoff.md
reports/edbt_eab/artifact_audit.md
reports/edbt_eab/blockers.md

artifacts/edbt_eab/mechanism_contract_matrix.csv
artifacts/edbt_eab/gold_standard_audit.json
artifacts/edbt_eab/baseline_matrix.csv
artifacts/edbt_eab/claim_evidence_matrix.csv
artifacts/edbt_eab/claim_evidence_matrix.json
artifacts/edbt_eab/release_manifest.json
```

若 P1 新实验经 gap analysis 后确有必要，还需提供相应的 pre-run protocol、raw cells、
analysis、manifest 和 tests。

---

## 15. 最终回报格式

最终消息必须按以下顺序报告，不能只说“完成”：

### 15.1 Restored vs rerun

- 哪些文件是恢复；
- 哪些实验是重跑；
- 恢复/重跑的原因；
- 环境、命令和 hashes。

### 15.2 Coverage

- canonical rows / expected rows；
- datasets / mechanisms / strengths / models / seeds；
- diagnostic rows/methods；
- governance rows/budgets；
- natural tasks/models/seeds；
- failures、duplicates、NaN、constant outputs。

### 15.3 Claims

用表格列出每个 claim：status、effect、CI、statistical unit、scope、source path、
main/supplement/exclude 建议。

### 15.4 EA&B gap closure

逐项说明：benchmark validity、baseline coverage、statistics、efficiency/scalability、
natural boundary、artifact 是否关闭；未关闭的必须给出 blocker。

### 15.5 Verification

原样给出：

- full pytest summary；
- release validator summary；
- artifact verifier summary；
- canonical SHA；
- paper claims SHA；
- artifact SHA。

### 15.6 Residual risks

列出所有仍不能进入论文的结果。不要用“基本完成”“大概率没问题”等表述。

---

## 16. 立即开始时的第一份回复

先做只读 inventory，然后回复：

1. 当前 git/worktree 状态；
2. M10 TabM 500-cell 产物是否在本地或远端找到；
3. official TabM 5,500-cell 主表及 manifest 是否通过 hash/code/environment 核验；
4. canonical builder 当前第一个真实 blocker；
5. 现有 statistics/natural/artifact 中哪些 hash 已因上游恢复而过期；
6. 你计划执行的最小工作序列；
7. 哪些 P1 实验需要先提交 protocol/gap analysis，而不是立即运行。

完成 inventory 后继续执行 P0 工作，不需要为了普通、可恢复问题等待确认。若需要改变
研究问题、引入新的外部数据、增加重大依赖、执行不可逆 Git 操作或显著扩大 GPU
预算，必须先停下并向用户申请授权。

