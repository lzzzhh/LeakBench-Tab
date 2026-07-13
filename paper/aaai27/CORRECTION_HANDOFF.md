# LeakBench-Tab 纠错复现完整任务书

## 1. 任务目标

在不新增模型、不新增污染机制、不扩展论文研究问题的前提下，修复会影响 AAAI-27 论文主结论的实现、数据血缘、统计推断和复现问题；重新运行受影响的既有实验单元；生成一套能够从原始结果表重建正文数字的 corrected evidence package。

这是一项 **纠错复现任务**，不是新实验阶段。最终结论必须服从修复后的数据，禁止为了保留旧 Finding 调整生成器或筛选结果。

工作目录：

```text
/Users/zhanhuilin/Downloads/RiskAgent
```

论文写作规范：

```text
paper/aaai27/WRITING_SPEC.md
paper/aaai27/NUMBERS.md
```

## 2. 强制边界

1. 不新增模型、机制、自然任务或研究问题。
2. 不新增内部 Phase 编号。
3. 只重跑因实现错误、数据缺失或统计输出缺失而受影响的既有矩阵。
4. 不覆盖旧结果；旧文件必须保留为 legacy evidence。
5. 新结果统一写入新的 corrected namespace，例如 `results/corrected_v1/`。
6. 不把 `leak_mask`、注入位置、`n_clean`、特征名中的 `leak`、机制标签或未来标签提供给 operational detector。
7. Oracle、policy-equivalent 和 operational metadata 必须分开实现、分开命名、分开报告。
8. 所有论文数字必须能由脚本从逐 cell 结果表重建。
9. 如果修复后旧结论不成立，更新结论，不得调参挽救。
10. 不修改 AAAI 官方 `aaai2027.sty`。

## 3. 已确认的问题清单

### 3.1 Critical：M08 Core 数据与标签无关

位置：`src/leakbench/mechanisms/__init__.py::_m08`

当前实现随机生成 entity rate 和 entity assignment，但二者均不依赖目标或具有实体层级目标结构。生成特征本质上是随机分组噪声，不能支持“entity leakage 无效但不可利用”的结论。

### 3.2 Critical：M09 Core 数据与标签无关

位置：`src/leakbench/mechanisms/__init__.py::_m09`

当前 source rate 与 source assignment 均为随机生成，与目标无关。它不是来源污染，只是随机来源噪声。

### 3.3 Critical：Meta M08 存在当前标签回流

位置：

- `experiments/leakbench/run_meta_tier.py`
- `experiments/leakbench/run_phase14.py`

当前实体均值使用全数据 `y`，包含当前样本自己的标签。该实现可能退化为 target encoding/self leakage，不能代表纯粹的未来实体生命周期聚合。

### 3.4 Critical：operational metadata 偷看污染身份

位置：`experiments/leakbench/run_operational_meta.py::compute_operational_scores`

已确认存在：

```python
if "leak" in name and j >= nc
if unique_ratio < 0.1 and j >= nc
```

这里使用特征名称和注入边界识别污染字段，因此所谓 operational A/S 并非真正 operational。由此产生的 A-RAW/A-DERIVED 增益不能继续作为论文证据，必须修复并重跑。

`experiments/leakbench/run_phase14.py::compute_scores` 中同样使用 `j >= nc`，也必须修复。

### 3.5 Critical：group governance 实际没有执行

位置：`experiments/leakbench/run_meta_tier.py::run_governance_strategy`

`groups` 保存整数索引，但执行代码用整数去匹配字符串 `feature_names`：

```python
if f in task["feature_names"]
```

这会令 `group_scores` 为空或分组移除不发生。当前 group recall 为 0 不能用于否定 group governance。

### 3.6 Critical：多种 governance 策略退化为同一策略

位置：`src/leakbench/governance/__init__.py`

当前 `GRAPH_CUT` 与 fixed field budget 使用相同的 top-k 字段移除逻辑；缺少 group metadata 时 group budget 又回退为 field budget。现有 `governance_v2.csv` 中 G3/G4/G6 大量逐行完全相同，不能被解释为三种独立治理方法的公平比较。

### 3.7 Critical：M10 ground truth 与生成内容冲突

位置：`src/leakbench/mechanisms/__init__.py`

- `mixed_legit` 实际由 `y + noise` 生成，并不 legitimate。
- injector 将所有新增列统一标成 `leak_mask=True`，无视返回的 `legitimate` 标签。

必须让 legitimate 分量来自预测时合法的 `X`，或者将其诚实标为污染；不能继续一边由 `y` 生成、一边标 legitimate。

### 3.8 High：M04 不是严格的未来/后结果窗口

`np.convolve(..., mode="same")` 是居中卷积，混合过去、当前和未来标签；当前标签也可能进入特征。必须明确预测时间方向并排除当前样本标签。

### 3.9 High：M05 使用未来值但配合随机切分

M05 定义了未来窗口，但 injector 默认 random split。时间污染应使用保序或显式时间切分，并处理尾部没有未来观测的样本，不能用固定零值制造额外信号。

### 3.10 High：Meta Tier 没有真正使用声明的 seed 复现

多个脚本声明 `SEEDS = [13, 42, 2026]`，实际注入调用固定 seed 42，输出行主要来自 method/strategy/budget 展开，而不是三个独立随机种子。

因此“4,032 experiment cells”的统计单位必须重新定义：它目前是 1,008 个 diagnostic rows 加 3,024 个 governance rows，而不是 4,032 个独立模型训练单元。

### 3.11 High：缺少可重建的核心逐 cell 结果

当前仓库存在 CatBoost 和 LightGBM 的逐 cell CSV，但没有找到 LR、RF、TabM 的完整逐 cell source tables。相关性、equal-weight aggregation 和 hierarchical bootstrap 不能仅从机制均值可靠恢复。

### 3.12 High：计划配置与实际矩阵不一致

`configs/paper/full_matrix.yaml` 写 5 seeds、27,500 cells；冻结 ledger 写 3 seeds、10,083 completed Core cells。必须区分 planned matrix 与 completed matrix，并生成 canonical completed manifest。

### 3.13 High：缺少论文要求的统计输出

尚未找到：

- hierarchical bootstrap confidence intervals；
- paired effect-size tables；
- Holm-corrected comparison families；
- 可重建 global/category-controlled correlation 的逐机制脚本和输出。

### 3.14 High：自然任务适配器存在静默 fallback 和文件选择问题

`benchmark_v2/datasets/adapters.py`：

- Lending Club 使用未排序的第一个匹配文件，可能错误读取 rejected 表；当前自然迁移结果为 8 个字段、0 个污染字段。
- Bank Marketing 只搜索目录顶层 CSV，但真实 CSV 位于子目录，导致静默使用 synthetic fallback。
- fallback 没有在结果表中留下明确 `is_synthetic` 标记。
- `_time_split` 仅按现有行序切分，没有显式按真实时间字段排序。

### 3.15 High：测试套件包含大量无效测试

大量测试仅执行 `assert True`，包括 claims、freeze、metadata、natural scope 等。这些测试不能验证实现或证据。

测试中还存在陈旧 M03 profile，与 corrected profile `C1-DH-XC` 冲突。

### 3.16 Medium：发布验证器使用错误解释器

`scripts/validate_release.py` 硬编码 `python3 -m pytest`，而当前工作环境的 pytest 安装在 Conda Python 中，导致 validator 为 20/21。应使用 `sys.executable`。

### 3.17 Medium：发布包不可独立复现

- `src/leakbench/cli/worker.py` 是 stub。
- release README 引用的 reproduction scripts 不存在。
- 根目录没有 dependency lock/requirements/pyproject。
- frozen release 只含少量 manifest/checksum 文件，未包含完整代码与结果。
- checksum 列表引用项目根目录文件，而非一个自包含发布包。

### 3.18 Medium：论文数字口径存在一处舍入冲突

`A-DERIVED delta = 0.094471`，标准三位小数应为 `+0.094`，不是 `+0.095`。审计脚本：

```text
paper/aaai27/source_data/audit_numbers.py
```

## 4. 必须采用的 M08/M09 修复定义

### 4.1 M08：未来实体结果聚合

生成带重复实体和时间的 panel 数据：

```text
u_e ~ Normal(0, sigma_entity^2)
y_e,t ~ Bernoulli(sigmoid(f(x_e,t) + u_e + epsilon_e,t))
```

污染字段：

```text
future_entity_rate(e,t)
  = shrinkage_mean({y_e,k: time_k > time_t})
```

要求：

1. 严格排除当前样本自己的 `y`。
2. 只允许使用严格未来的同实体结果。
3. 未来样本不足时使用全局 prior，不得填固定零。
4. strength 通过未来聚合权重/噪声控制，不能通过是否包含当前标签控制。
5. 至少 50 个实体，每实体建议至少 20 条记录；记录有效未来样本数。
6. strict 与 permissive 使用相同测试行和模型配置，只在污染字段是否存在上不同。
7. hierarchical bootstrap 在 dataset/seed 之下按 entity cluster 重采样。

### 4.2 M09：结果依赖的采集来源偏移

先生成原始 `X, y`，再生成部署时不可获得的 collection source：

```text
P(source=s | y=1) proportional to exp(alpha * delta_s)
P(source=s | y=0) proportional to exp(-alpha * delta_s)
```

要求：

1. source 表示数据采集/结果处理阶段的来源，声明为预测时不可用。
2. 使用原始 categorical source ID 或完整 one-hot group，不使用有序整数作为连续数值。
3. 每个 seed 随机置换 source label，排除编码顺序效应。
4. strength 通过类别条件来源分布的 separation 控制，记录 TV/JS divergence。
5. 不使用 full-data source target mean；否则会退化成 target encoding。
6. 至少 8 个来源，保证每个来源/类别有足够样本；不足时该 cell 标记 invalid，不得静默运行。
7. hierarchical bootstrap 在 dataset/seed 之下按 source cluster 重采样。

### 4.3 不允许的“修复”

- 不得将 M08/M09 直接改成 `y + noise`。
- 不得调 strength 直到重新得到旧 `DL-XL` profile。
- 不得把当前样本标签放入实体/来源聚合。
- 不得根据结果选择更好看的 seed、数据集或模型。

## 5. 执行顺序

### Step 1：冻结旧证据与建立 inventory

1. 记录所有现有代码、配置、结果文件的 SHA256。
2. 建立 legacy manifest，禁止覆盖旧 CSV。
3. 列出每个结果表的行数、列、dataset/mechanism/strength/model/seed 覆盖。
4. 明确哪些逐 cell 数据存在，哪些必须按原协议重跑。
5. 输出 `reports/corrected_v1_inventory.md`。

### Step 2：先写会失败的行为测试

至少覆盖：

- M08 特征与 entity/time metadata 完整，且不含当前标签。
- 打乱 future entity outcomes 后 M08 信号消失。
- 打乱 source/label 对应后 M09 信号消失。
- M09 source label permutation 不改变指标分布。
- M10 legitimate mask 与 mechanism label 一致。
- M04/M05 不读取当前标签。
- operational score 对列顺序、特征匿名化保持一致。
- operational score 代码路径不读取 `leak_mask/n_clean/mechanism label`。
- group strategy 实际移除整个 group。
- graph strategy 与 field strategy 在非退化图上产生不同 mask。
- adapters 使用确定的真实文件，并显式报告 lineage/fallback。

先观察测试失败，再修改实现。

### Step 3：统一机制生成接口

机制生成器需要接收并返回足够的结构信息：

```text
X, y, timestamps, entity_ids, source_ids,
feature_names, feature_roles, feature_availability,
groups, graph_edges, leak_mask, split indices
```

不要继续让 `_inject(cfg, y, n)` 在缺少 `X/time/entity/source` 的情况下伪造结构化机制。保持修改最小，但接口必须能表达 M08/M09/M10 的真实语义。

### Step 4：修复 M04/M05/M08/M09/M10

按本任务书定义执行。为每个机制加入构造不变量和 negative-control tests。

### Step 5：彻底隔离 operational 与 oracle metadata

创建独立输入结构：

```text
OperationalMetadata
OracleMetadata
PolicyEquivalentMetadata
```

operational detector 只能读取在部署审计中合理可见的 schema/lineage/statistical fields。禁止通过位置、名称或注入顺序识别污染。

增加 feature permutation test：随机打乱列和匿名化字段名后，将评分映射回原 ID，结果应保持一致。

### Step 6：修复 governance 比较

1. 统一 group membership 表示，建议全部使用 stable feature IDs。
2. 验证 group removal 真的移除组内全部字段。
3. graph cut 必须使用图邻接/连通结构；若无法实现独立策略，应从主比较中删除或明确标为等价 baseline，不能冒充独立方法。
4. lifecycle removal 只能使用 operational lifecycle metadata。
5. 策略比较必须同时报告 contaminated-feature recall、legitimate retention、实际移除字段数和 downstream utility。
6. 不同策略的预算要按可比资源定义；至少同时给 field count 和 review unit count，避免按 group 百分比与 field 百分比直接比较。

### Step 7：修复自然任务适配器

1. Lending Club 明确选择 accepted 数据；不得依赖 glob 顺序。
2. Bank Marketing 递归找到 `bank-additional-full.csv`。
3. 默认禁止静默 synthetic fallback；缺文件时明确失败。若测试需要 synthetic，必须显式参数启用并写入 lineage。
4. 每个 NaturalTask 返回 source path、license、is_synthetic、target definition、prediction time、split rule。
5. 时间任务先按时间字段排序再切分。
6. Lending Club 若仍无可审计污染字段，保持 adapter-limited 并排除 transfer claim。

### Step 8：恢复 canonical completed matrix

1. 从现有逐 cell 文件恢复能恢复的 LR/RF/CatBoost/LightGBM/TabM 数据。
2. 如果 LR/RF/TabM 原始逐 cell 数据确实丢失，只重跑原 Core 协议中缺失的部分。
3. 对受机制修复影响的 M04/M05/M08/M09/M10，重跑原有模型/数据集/strength/seed 覆盖。
4. 不补跑 TabPFN，不扩大模型覆盖。
5. 每个 cell 保存 run ID、dataset、mechanism、strength、model、seed、strict metric、permissive metric、aligned harm、diagnostic metrics、status、runtime、config hash。
6. 失败 cell 不得删除，必须标记 failure type。

### Step 9：统计推断

在 corrected source tables 上生成：

1. dataset-level paired effects；
2. equal-weight five-core-model aggregation；
3. hierarchical bootstrap 95% CI；
4. mechanism/category/model comparison effect sizes；
5. comparison-family 内 Holm correction；
6. global Spearman correlation；
7. category-controlled incremental effect；
8. excluding-simple 和 within-structured sensitivity；
9. 自然任务 rank、Top-K、MRR，避免单正类时只报告 AUPRC。

预先固定 bootstrap seed 和重复次数。统计脚本必须只读取 corrected CSV，不得调用训练代码。

### Step 10：替换无效测试与修复 validator

1. 将影响论文证据的 `assert True` 替换为真实行为或结果表断言。
2. 删除/更新陈旧 M03 profile 常量。
3. validator 使用 `sys.executable -m pytest`。
4. validator 必须检查 required source tables、manifest、statistics、paper number audit，而不仅是文件存在。

### Step 11：重建复现与发布包

至少补齐：

- `pyproject.toml` 或锁定依赖文件；
- 一条从 frozen/corrected tables 重建报告与图表的命令；
- 一条受影响实验的重跑命令；
- corrected manifest；
- checksums；
- 数据来源与许可证说明；
- 匿名化 code archive；
- 不含 stub 的必要执行入口。

### Step 12：更新论文证据，不直接润色结论

1. 先更新 `paper/aaai27/NUMBERS.md`。
2. 每个数字写明 source CSV、aggregation function 和 raw precision。
3. 生成 old vs corrected claim-impact table。
4. 只有完成数字审计后才能更新 `main.tex` Results。
5. 保留所有修复后 refuted 或 weakened findings。

## 6. 验收标准

任务只有同时满足以下条件才算完成：

1. M08/M09 生成器具有明确结构语义和预测时间无效性。
2. M08 不包含当前标签；M09 不使用全数据 target-rate encoding。
3. operational code 路径不包含基于 `leak` 名称、`j >= n_clean`、leak mask 或机制 ID 的分支。
4. feature permutation/anonymization 测试通过。
5. group governance 在最小合成例上实际移除完整组。
6. G3/G4/G6 不再因为代码复用而逐行机械相同；若科学上等价，必须明确证明并标注。
7. Meta 脚本真正循环所有声明 seeds，或删除虚假的 seed 声明并诚实重定义统计单位。
8. 自然任务不发生未记录的 synthetic fallback。
9. 所有正文 headline 数字由只读统计脚本重建。
10. hierarchical bootstrap、paired effects 和 Holm 输出存在且可复现。
11. 旧结果未被覆盖，corrected results 有独立 manifest/checksum。
12. 连续运行两次数字审计产生相同 hash。
13. 测试与 release validator 在同一解释器环境全部通过。
14. 最终报告明确列出哪些原论文 Finding 被支持、削弱、推翻或暂时无法判断。

## 7. 最终交付物

必须交付：

```text
reports/corrected_v1_inventory.md
reports/corrected_v1_root_cause.md
reports/corrected_v1_claim_impact.md
results/corrected_v1/canonical_cells.csv
results/corrected_v1/statistics/*.csv
results/corrected_v1/manifest.json
results/corrected_v1/SHA256SUMS
scripts/reproduce_corrected_results.sh
scripts/audit_paper_numbers.py
paper/aaai27/NUMBERS.md
```

并在最终回复中报告：

1. 修改了哪些文件；
2. 重跑了哪些既有实验单元；
3. 运行命令及真实通过结果；
4. 哪些旧数字发生变化；
5. 哪些结论被支持、削弱或推翻；
6. 仍未解决的阻断项。

## 8. 最终执行指令（可直接复制给另一个 Agent）

```text
请在 /Users/zhanhuilin/Downloads/RiskAgent 中执行 LeakBench-Tab 的纠错复现。

首先完整阅读：
1. paper/aaai27/CORRECTION_HANDOFF.md
2. paper/aaai27/WRITING_SPEC.md
3. paper/aaai27/NUMBERS.md
4. configs/leakbench/mechanism_registry.yaml
5. reports/phase10r_corrected_results.md
6. reports/claim_evidence_matrix_final.md

严格遵守 CORRECTION_HANDOFF.md 的任务边界、执行顺序、M08/M09 定义、验收标准和交付物要求。

这是纠错复现，不是新增实验：禁止新增模型、机制、自然任务、研究问题或 Phase；禁止覆盖旧结果；新输出统一写到 results/corrected_v1/。先建立旧证据 inventory 和 checksum，再写能够复现当前失败的行为测试，然后修复实现。所有受影响的既有实验可以重跑，但不得扩展原覆盖范围。

特别注意以下四个最高优先级错误：
1. M08/M09 Core 当前与 y 没有有效结构关系；
2. operational metadata 使用 leak 名称和 j>=n_clean 偷看污染身份；
3. group governance 把整数索引当特征名，导致策略不执行；
4. M10 legitimate/leakage 内容与 ground-truth mask 冲突。

不要保留预设结论。修复后的数据决定 M08/M09 profile、metadata gain 和 governance verdict；如果旧 Finding 被推翻，必须在 corrected_v1_claim_impact.md 和 paper/aaai27/NUMBERS.md 中诚实更新。

持续执行到所有可修复项完成、受影响矩阵重跑、统计输出生成、数字审计可重复、测试和 validator 通过。若原始逐 cell 数据缺失，按原冻结协议仅重跑缺失/受影响部分，并记录 run manifest；不得从汇总均值伪造置信区间。
```
