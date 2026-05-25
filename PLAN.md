# SecMCP — 基于任务漂移的 Agent Prompt Injection 激活检测器计划

## Context

当前项目目标是把原 TaskTracker 中“激活值差异分类器”迁移到 agent security 场景。旧方案直接对整段 agent context 提取 last-token activation，再训练 RF/logistic 二分类器；在 Mistral 上测试 AUROC 约 0.6，说明长 agent context 会稀释局部注入信号。

新方案改为 **Task-Drift Detection**：在每次 tool response 返回后、下一次 LLM 决策前，比较模型内部状态是否从原始用户任务发生异常漂移。

核心插入点：

```python
ToolsExecutionLoop([ToolsExecutor(), SecMCPTaskDriftDetector(...), llm])
```

AgentDojo 已支持这种 pipeline 结构，因此可以正确评估 detector 对 `utility/security` 的影响。

## 目标仓库结构

```text
SecMCP/
├── PLAN.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── models.yaml
│   ├── data.yaml
│   ├── training.yaml
│   └── eval.yaml
├── data/                         # 本地 benchmark 数据，只读，不入 git
│   ├── InjecAgent/
│   ├── AgentHarm/
│   ├── ASB/
│   ├── AgentTraj-L/
│   └── AgentDojo/
├── src/secmcp/
│   ├── data/                     # schema、loader、split
│   ├── models/                   # HF 模型加载、chat template、truncation、hook
│   ├── activations/              # whole-context baseline + task-drift activation
│   ├── detectors/                # task_drift、RF/logistic baseline、metrics
│   ├── integrations/             # AgentDojo / ASB runtime hook
│   └── baselines/                # PromptGuard、spotlighting、PI detector 等
├── scripts/
│   ├── 01_unify_datasets.py
│   ├── 02_extract_activations.py
│   ├── 03_train_detector.py
│   ├── 04_eval_agentdojo.py
│   ├── 05_eval_asb.py
│   ├── 06_run_baselines.py
│   └── 07_summarize_results.py
├── tests/
└── outputs/                      # split、activation、detector、eval 输出，不入 git
```

## Phase 0 — 仓库与配置

目标：把 `SecMCP/` 建成独立、可测试、可复现实验仓库。

- 在 `SecMCP/` 初始化独立 git 仓库；父目录是另一个 git 仓库不影响内部提交。
- `.gitignore` 忽略 `/data/`、`outputs/` 生成内容、缓存和本地 agent 状态。
- `configs/models.yaml` 固定 4 个模型：
  - `mistral_7b_v03`：第一优先实验模型，成本最低，先验证方案；本地路径 `/hub/huggingface/models/MistralAI/Mistral-7B-Instruct-v0.3`。
  - `gemma2_9b`：第二优先，验证跨模型稳定性；本地路径 `/hub/huggingface/models/google/gemma-2-9b-it`。
  - `qwen3_32b`：中等规模泛化实验；本地路径 `/hub/huggingface/models/Qwen3-32B`。
  - `llama3_3_70b`：最终大模型验证；本地路径 `/hub/huggingface/models/Llama-3.3-70B-Instruct`。
- 统一使用 `head_tail` 截断；训练和推理必须共享同一 tokenizer、chat template、truncation、layers。

验收：配置可加载；模型层号、hidden dim、max tokens 自洽；测试通过。

## 模型路径约定

后续实验统一按以下本地 Hugging Face checkpoint 路径加载模型；如果运行时报 `HFValidationError` 并把绝对路径当成 repo id 校验，优先检查当前执行环境是否能访问 `/hub/huggingface/models`。

```text
llama3.3-70B: /hub/huggingface/models/Llama-3.3-70B-Instruct
Qwen3-32B:    /hub/huggingface/models/Qwen3-32B
mistral3-7B:  /hub/huggingface/models/MistralAI/Mistral-7B-Instruct-v0.3
gemma2-9B:    /hub/huggingface/models/google/gemma-2-9b-it
```

## Phase 1 — 数据统一与 Trajectory 保留

目标：loader 输出 agent-style trajectory，而不是孤立文本分类样本。

统一 schema：

```python
UnifiedSample(
    label: int,
    text: str,
    messages: list[dict],
    tools: list[dict] | None,
    source: str,
    kind: str,
    sample_type: str,
    metadata: dict,
)
```

数据集用途：

- `AgentDojo`：主评估 benchmark；直接保留源 JSON `messages`，20% held-out test。
- `ASB`：辅助评估 benchmark；构造 system + user task + assistant tool call + tool response。
- `InjecAgent`：训练和诊断；恶意指令必须放入 tool response。
- `AgentHarm`：补充 harmful user intent，不作为 tool-injection 主指标。
- `AgentTraj-L`：benign agent trajectory，按 domain 采样，避免 benign 被单一数据源支配。

划分策略：

- AgentDojo 按 `(user_task_id, injection_task_id)` 或 injection hash 做 group split，避免泄漏。
- 其它数据按 `(source, label)` 分层进入 train/val。
- `train/val/test` 输出到 `outputs/splits/*.jsonl`。

注意点：不能只拼字段文本，否则检测器会学成文本分类器，不是 agent context 漂移检测器。

## Phase 2 — 模型加载、Chat Template 与截断

目标：所有模型通过统一接口提取指定层 hidden states。

- `load_shared_model(model_name)` 进程内缓存模型和 tokenizer。
- `last_token_hidden_states(model, tokenizer, messages, layers, cfg)` 返回 `[n_layers, hidden_dim]`。
- 对 Mistral/Gemma 这类不支持 system role 的模板，把 system 合并到 user。
- 对 tool-role 模板不兼容情况，允许 fallback 到稳定的 role/text 渲染，但训练和推理必须一致。
- 长上下文用 `head_tail`：保留 system + 原始 task 头部，以及最新 observation 尾部。

验收：mock tokenizer/model 单测通过；真实模型抽样可跑通一个短样本。

## Phase 3 — Task-Drift Activation Extraction

目标：从“一条 trajectory 一个 activation”改成“一条 trajectory 多个 tool-step activation”。

对每个 `role == "tool"` 消息构造一个检测点：

- `task_prefix = system + first user task`
- `history_prefix = 当前 tool response 之前的完整上下文`
- `post_tool_prefix = 当前 tool response 之后的完整上下文`

分别提取：

- `task_anchor`
- `history_anchor`
- `post_tool_state`

输出到：

```text
outputs/drift_activations/{model}/{split}/
├── drift_00000.pt     # task/history/post tensors
├── labels_00000.pt
└── meta_00000.jsonl
```

meta 至少包含：

- `sample_index`
- `step_index`
- `tool_message_index`
- `source`
- `kind`
- `sample_type`
- `tool_name`
- 原始 metadata

注意点：

- 检测点必须发生在 tool response 后、下一次 assistant 决策前。
- 如果一个 trajectory 没有 tool response，不进入 task-drift 训练；可保留给 whole-context baseline。
- 如果未来有精确 injection span/turn 标注，优先用 step-level label；没有则使用 trajectory label 的 weak supervision。

## Phase 4 — Drift Feature 与检测模型

目标：用漂移特征训练主 detector。

核心特征：

- `global_drift = post_tool_state - task_anchor`
- `incremental_drift = post_tool_state - history_anchor`
- `history_progress = history_anchor - task_anchor`
- `relative_drift = ||incremental_drift|| / (||history_progress|| + eps)`
- 每层 L2 norm、cosine similarity
- 与 benign incremental/global drift anchors 的 mean/min/max/std 距离

默认主模型：

- `HistGradientBoostingClassifier`
- validation set 选择 threshold，主校准目标为 FPR@95TPR。
- trajectory-level score 默认从 step-level score 聚合；当前默认是 `first_exceed_3`（前 3 步内最高分），可选 `max / top2_mean / mean / first_exceed_K / cusum[_wN_sX_rY]`。早期方案默认 `max`，但 `max` 在长 benign 轨迹上有 1−(1−p)ⁿ 的 FPR 放大问题，参见下方"方法学修复"段。

保留 baseline：

- `rf_anchor`
- `logistic_diff`
- whole-context last-token detector

注意点：

- RF 只作为 baseline，不作为主方案。
- 不急于上 MLP/contrastive learning；先确认 drift feature 是否显著优于 whole-context。
- 若后续 clean/malicious pair 构造稳定，再加入 paired contrastive objective。

## Phase 5 — AgentDojo 集成评估

目标：在真实 AgentDojo pipeline 中评估 detector 对安全性和可用性的影响。

集成方式：

```python
tools_loop = ToolsExecutionLoop([
    ToolsExecutor(),
    SecMCPTaskDriftDetector(
        detector_path="outputs/detectors/{model}/task_drift_best.pkl",
        model_name="{model}",
        raise_on_detection=True,
    ),
    llm,
])
```

运行模式：

- `raise_on_detection=False`：只记录 score，用于离线调阈值和分析。
- `raise_on_detection=True`：触发 `AbortAgentError`，阻止下一次 LLM 决策。

AgentDojo 指标：

- `utility`
- `security`
- attack success rate
- detector trigger rate
- benign false abort rate
- per-step score、trigger turn、tool name

主 benchmark：

- AgentDojo workspace / slack / travel / banking suites。
- clean `none/none.json` 和 `important_instructions/injection_task_*.json` 必须都跑。

注意点：

- AgentDojo 是主评估，不应只看离线 AUROC。
- detector 触发太激进会提升 security 但损害 utility，必须同时报告。

## Phase 6 — ASB 与 Baseline 对比

目标：验证方案不只对 AgentDojo 有效。

ASB 评估：

- 使用 ASB agent task、normal tools、attack tools 构造完整 agent context。
- 检测点同样放在 tool response 后。
- 报告 ASB 上的 AUROC/AUPRC/FPR@95TPR，以及按 agent role / attack type 分组结果。

Baseline：

- PromptGuard / text classifier
- AgentDojo built-in `TransformersBasedPIDetector`
- spotlighting / delimiting
- repeat user prompt / sandwiching
- old whole-context activation detector

注意点：

- 文本 baseline 检测的是 tool text；SecMCP 检测的是 model state drift，比较时要同时报告 runtime 成本。
- ASB 的攻击工具描述不能作为孤立 malicious text 训练，必须嵌入 tool response。

## Phase 7 — 汇总、消融与报告

目标：明确 task-drift 是否解决旧方案 AUROC 低的问题。

必须做的消融：

- whole-context baseline vs task_drift
- task-only drift vs history-only drift vs combined drift
- `max` vs `top-2 mean` trajectory aggregation
- global benign anchors vs source-normalized benign anchors
- 不同层组合：浅层 / 中层 / 深层 / concat
- Mistral vs Gemma vs Qwen vs Llama

报告文件：

```text
outputs/eval/{benchmark}/{model}/metrics.json
outputs/eval/{benchmark}/{model}/step_scores.jsonl
outputs/eval/summary.csv
```

核心指标：

- AUROC
- AUPRC
- FPR@95TPR
- AgentDojo utility/security
- benign false abort rate
- malicious missed-detection rate
- per-source/per-suite breakdown

成功标准：

- Mistral task-drift 在 held-out test 上明显优于 whole-context baseline。
- AgentDojo 上 security 提升时，utility 不出现不可接受下降。
- 至少在 AgentDojo 和 ASB 两个 benchmark 上趋势一致。

## 当前默认实验顺序

1. 先跑 `mistral_7b_v03`，因为已有激活经验且成本最低。
2. 用 AgentDojo held-out test 作为主结论。
3. 用 ASB 做外部验证。
4. 再扩展到 `gemma2_9b`、`qwen3_32b`。
5. 最后只在关键设置上跑 `llama3_3_70b`。

## 关键风险与注意事项

- 长上下文截断必须完全一致，否则训练/推理分布漂移。
- AgentDojo split 必须 group-disjoint，避免同一 injection 文本泄漏。
- 恶意样本必须在 agent context 中体现，不允许把 attacker instruction 当独立短文本分类。
- task-drift detector 应在 tool response 后立即运行，不能等 assistant 已经执行下一步后再检测。
- benign tool outputs 也可能造成大漂移，因此阈值必须用 validation set 校准。
- 只看 AUROC 不够，必须同时看 FPR@95TPR 和 AgentDojo utility/security。

## 方案演进简短总结

之前的方案延续了 TaskTracker 里用于短上下文 prompt injection / RAG 场景的激活值比较思路：把整段输入上下文渲染成文本，对模型指定层提取 last-token activation，再用 benign anchor 距离、activation diff、RF 或 logistic 分类器做二分类。这种实现方式在短 prompt 或任务边界清晰的输入上可行，但迁移到 agent 场景后，agent context 往往包含 system prompt、原始任务、多轮 assistant/tool 交互和较长 observation；恶意注入只占其中很小一段，整段 context 的单个 last-token 表示容易被正常上下文稀释，因此 Mistral 测试集 AUROC 只有约 0.6，效果不理想。

新方案改为基于 **任务漂移** 的检测：不再判断整段 context 是否异常，而是在每次 tool response 返回后、下一次 LLM 决策前，比较模型状态相对原始用户任务和历史上下文是否发生异常漂移。实现上，数据层保留结构化 `messages`；激活提取层为每个 tool step 构造 `task_prefix`、`history_prefix`、`post_tool_prefix`，分别提取 `task_anchor`、`history_anchor`、`post_tool_state`；训练层使用 `global_drift`、`incremental_drift`、`relative_drift` 和 benign drift anchor 距离作为特征，默认用 `HistGradientBoostingClassifier` 训练 `task_drift` detector；评估时把 `SecMCPTaskDriftDetector` 插入 AgentDojo 的 `ToolsExecutionLoop([ToolsExecutor(), detector, llm])`，确保检测发生在 tool output 进入上下文之后、agent 继续行动之前。

## 当前进度对照

以下状态来自当前代码和本地 `outputs/` 产物的静态检查，适合作为继续实验前的工作底稿。

### 已经较完整落地

- Phase 0 的仓库骨架、配置文件、输出目录约定和测试入口已经具备。`configs/models.yaml`、`configs/data.yaml`、`configs/training.yaml`、`configs/eval.yaml` 均可加载，基础单测覆盖了配置、schema、截断、loader、activation dataset、detector metric 等路径。
- Phase 1 的统一 schema、数据源 loader、合并脚本和 split 脚本已经存在。`UnifiedSample` 支持 `messages`、`tools`、`metadata`，AgentDojo 按 `split_group` 做 held-out，其他来源按 `(source, label)` 分层；`scripts/01_unify_datasets.py` 能写出 `outputs/splits/{train,val,test}.jsonl`。
- Phase 2 的模型加载、chat message 规范化、system-role 兼容处理、head-tail 截断和 last-token hidden-state hook 已经实现，并有 mock 单测。训练和推理共享 `load_shared_model`、模型配置层号和截断配置。
- Phase 3 的 task-drift 激活提取代码已经实现：每个 `role == "tool"` 的消息会生成 `task_prefix`、`history_prefix`、`post_tool_prefix`，输出 `drift_*.pt`、`labels_*.pt`、`meta_*.jsonl` 到 `outputs/drift_activations/{model}/{split}/`。脚本 `scripts/02_extract_activations.py` 当前默认走 `task_drift` 模式，旧 whole-context 仍可通过 `--mode whole_context` 显式运行；task-drift 路径已有 tqdm step 进度条和 `--log-every` 心跳输出。
- Phase 4 的主 detector 已经可训练：`HistGradientBoostingClassifier`、benign incremental/global drift anchors、`include_anchor_distances` 配置、每层 norm/cosine、trajectory-level 聚合（默认 `first_exceed_3`，可选 `max / top2_mean / mean / cusum[_wN_sX_rY]`）、validation threshold 选择、metrics 保存都已经在代码中。旧 `rf_anchor` 和 `logistic_diff` whole-context baseline 训练代码也保留；训练入口已有 shard 加载、特征构造、fit、val/test scoring、保存阶段的命令行进度输出，可用 `--no-progress` 关闭。
- Phase 5 的 AgentDojo pipeline element 已有集成类 `SecMCPTaskDriftDetector`。它能在 tool message 后计算当前 step score，按 detector 保存的 aggregation 聚合成 trajectory score，并在 `raise_on_detection=True` 时抛出 `AbortAgentError`。

### 仍然只是部分具备

- Phase 1 的“保留结构化 trajectory”在代码路径上具备，且当前本地 `outputs/splits/*.jsonl` 已经可以用于 task-drift：`train` 有 24,541 samples / 84,513 tool steps，`val` 有 6,135 samples / 21,158 tool steps，`test` 有 6,424 samples / 25,163 tool steps。
- Phase 3 的实现依赖样本中真实存在 `role == "tool"` 消息。ASB 和 InjecAgent loader 会构造 tool message；AgentDojo loader 会保留 run JSON 中的 messages，但需要确认原始 AgentDojo run 文件里是否真的采用 tool role，或是否需要把 AgentDojo 的 observation / tool-result 角色映射到 `tool`。
- Phase 4 目前实现的是 combined drift 主特征。PLAN 中的 task-only drift、history-only drift、combined drift 消融还没有显式配置入口或脚本化 sweep；source-normalized benign anchors 也还没有实现。
- Layer ablation 主要依赖 `configs/training.yaml` 的 `layers.mode` 和模型配置层号手工切换；浅层 / 中层 / 深层 / concat 的系统化实验脚本还没有写。
- Whole-context baseline 的 activation 和 `rf_anchor` / `logistic_diff` 训练路径存在；但它们与 task-drift 的统一比较、统一表格和报告输出还没有脚本化。

### 还没有真正落地

- Phase 5 的完整 AgentDojo benchmark runner 尚未实现。仓库中没有 `scripts/04_eval_agentdojo.py`，因此还不能自动跑 workspace / slack / travel / banking suites，也还没有自动产出 utility、security、attack success rate、detector trigger rate、benign false abort rate、trigger turn、tool name 等报告文件。
- Phase 6 的 ASB evaluation runner 尚未实现。仓库中没有 `scripts/05_eval_asb.py`，也没有按 agent role / attack type 分组的 ASB 指标输出。
- Baseline runner 尚未实现。`src/secmcp/baselines/` 目前只有包占位，没有 PromptGuard、AgentDojo built-in PI detector、spotlighting / delimiting、repeat prompt / sandwiching 的统一运行脚本。
- Phase 7 的汇总脚本尚未实现。仓库中没有 `scripts/07_summarize_results.py`，`outputs/eval/{benchmark}/{model}/metrics.json`、`step_scores.jsonl`、`outputs/eval/summary.csv` 还没有生成链路。
- `qwen3_32b`、`llama3_3_70b` 的实际激活提取、detector 训练和跨模型比较目前没有本地产物。

### 当前本地产物含义

- `outputs/splits/{train,val,test}.jsonl` 已存在，且当前读回后可以产生 task-drift tool steps，可以直接作为激活提取输入。
- `outputs/activations/{mistral_7b_v03,gemma2_9b}/{train,val,test}/` 已存在，属于旧 whole-context baseline 激活；它们不能替代 `outputs/drift_activations/`。
- `outputs/drift_activations/mistral_7b_v03/{train,val,test}/` 已存在，属于新 task-drift 主方案激活产物：`train` 为 846 shards / 84,513 steps，`val` 为 212 shards / 21,158 steps，`test` 为 252 shards / 25,163 steps。
- `outputs/detectors/mistral_7b_v03/rf_anchor_best.pkl` 已存在，属于旧 whole-context RF baseline，不是 task-drift detector。
- `outputs/detectors/mistral_7b_v03/task_drift_best.pkl` 和 `task_drift_metrics.json` 已存在，属于 Mistral task-drift detector。

### 当前可直接运行的命令

生成 task-drift 激活，默认有进度显示；`--shard-size 100` 时 shard 数等于 `ceil(tool_steps / 100)`，文件从 `00000` 开始编号，所以最后一个 `00211` 代表第 212 个 shard。

```bash
conda run -n taskdrift python scripts/02_extract_activations.py \
  --model mistral_7b_v03 --split train --mode task_drift --shard-size 100 --log-every 1

conda run -n taskdrift python scripts/02_extract_activations.py \
  --model mistral_7b_v03 --split val --mode task_drift --shard-size 100 --log-every 1

conda run -n taskdrift python scripts/02_extract_activations.py \
  --model mistral_7b_v03 --split test --mode task_drift --shard-size 100 --log-every 1
```

训练 Mistral task-drift detector：

```bash
conda run -n taskdrift python scripts/03_train_detector.py \
  --model mistral_7b_v03 --detector task_drift
```

训练时会读取 `outputs/drift_activations/{model}/{train,val,test}/`，输出 `outputs/detectors/{model}/task_drift_best.pkl` 和 `task_drift_metrics.json`。如果只想先用 train/val 训练并跳过 test 评估，加 `--no-test`；如果想关闭加载和训练阶段的进度输出，加 `--no-progress`。

快速验证当前 shard 是否完整：

```bash
python -c 'from pathlib import Path
root=Path("outputs/drift_activations/mistral_7b_v03")
for split in ["train","val","test"]:
    out=root/split
    metas=sorted(out.glob("meta_*.jsonl"))
    print(split, len(list(out.glob("drift_*.pt"))), len(list(out.glob("labels_*.pt"))), len(metas), sum(1 for p in metas for _ in p.open()))'
```

### 最近下一步

Mistral 的 task-drift 激活和 detector 已经具备；下一步应先检查 `task_drift_metrics.json` 的 val/test AUROC、AUPRC、FPR@95TPR，再决定是否跑 Gemma 的同一流程。AgentDojo / ASB runtime 评估和 baseline 对比仍需要后续补脚本。

## 改进建议与执行记录

当前 Mistral detector 的主要问题不是漏检，而是 benign false abort 过高。原始 detector 在 held-out AgentDojo test 上约为 `TPR=95.0% / FNR=5.0% / FPR=69.1%`；这不能直接换算为安全 benchmark ASR，最多只能作为 detector-only 的漏检上界参考。真实 ASR 必须在 AgentDojo / ASB runtime 中统计“攻击成功且未被 abort”的比例。

### 已落地的直接改进

- 阈值校准改为可约束 false abort：`task_drift.threshold_max_fpr` 默认设为 `0.20`，`scripts/03_train_detector.py` 支持 `--threshold-target-tpr` 和 `--threshold-max-fpr`。metrics 中会保存当前阈值下的 TP/FP/TN/FN、TPR/FNR/FPR/TNR，以及多个 target-FPR 的 tradeoff。
- 训练加入 step-level label imbalance 处理：`task_drift.sample_weight: balanced`，也可用 `--sample-weight none|balanced` 覆盖。
- AgentDojo split 改为 `train/val/test` 三方 `split_group` 互斥，避免 train 与 val 共享同一 injection/task group，从而让 val threshold calibration 更接近 held-out 行为。需要重新运行 `scripts/01_unify_datasets.py`、重新抽取 task-drift 激活、重新训练 detector 后才会反映到正式结果。
- 训练 metrics 增加分组诊断：`diagnostics.{val,test}_group_diagnostics` 按 `source`、`sample_type`、`metadata.suite_name` 输出混淆矩阵，便于定位 hard negatives 和 suite-level false abort。
- Step-level label 已做第一版修正：ASB / InjecAgent 用显式恶意 tool index；AgentDojo 用 injection 文本片段匹配 tool output。命中的 tool step 标 `1`，同一恶意 trajectory 中未命中的 tool step 标 `0`；无法定位时回退 trajectory label。

### 需要重跑的命令

由于 split 逻辑已改变，旧 `outputs/splits/` 和旧 activation shards 不再代表当前代码。完整重跑顺序：

```bash
conda run -n taskdrift python scripts/01_unify_datasets.py

conda run -n taskdrift python scripts/02_extract_activations.py \
  --model mistral_7b_v03 --split train --mode task_drift --shard-size 100 --log-every 1 --no-resume
conda run -n taskdrift python scripts/02_extract_activations.py \
  --model mistral_7b_v03 --split val --mode task_drift --shard-size 100 --log-every 1 --no-resume
conda run -n taskdrift python scripts/02_extract_activations.py \
  --model mistral_7b_v03 --split test --mode task_drift --shard-size 100 --log-every 1 --no-resume

conda run -n taskdrift python scripts/03_train_detector.py \
  --model mistral_7b_v03 --detector task_drift
```

如需比较更保守或更激进的 utility/safety tradeoff：

```bash
conda run -n taskdrift python scripts/03_train_detector.py \
  --model mistral_7b_v03 --detector task_drift --threshold-max-fpr 0.10

conda run -n taskdrift python scripts/03_train_detector.py \
  --model mistral_7b_v03 --detector task_drift --threshold-max-fpr 0.30
```

### 方法学修复（已落地，Mistral 需重抽激活）

- **弱标签前向传播**：`tool_steps_for_sample` 不再只把命中 injection 文本片段的 tool step 标 1。命中第一个 injection step 起，trajectory 之后所有 tool step 都标 1。原先的实现会把"被劫持但非匹配"的后续 tool step 标 0，使分类器对同一类 hijacked-state 表征收到自相矛盾的监督。`step_label_source` 仍标记为 `matched_injection_tool` 以与 trajectory-level fallback 区分。
- **task-anchored 激活提取**：原 `task_anchor / history_anchor / post_tool_state` 取的是三段不同前缀的 *last-token* hidden state，等价于"两个不同 token 在同一空间的差"，不是对 task 的漂移度量。新实现把原始 user-task 文本在每段前缀末尾再次拼接（`TASK_ANCHOR_SEPARATOR + task_text` 作为 raw token），causal attention 让重读的 task token 看到全部前文，最后对 task token 位置做 mean pooling。三段前缀拼接同样的 task，差异即"加入这段历史/工具响应后，模型对 task 的理解漂移了多少"，与 TaskTracker 原始设定一致。
  - 新 hook：`secmcp.models.hooks.task_anchored_hidden_states(model, tokenizer, prefix_messages, task_text, layers, cfg)`。
  - 旧 `last_token_hidden_states` 仍保留，仅供 whole-context baseline 与回归测试使用。
  - 每条 meta 记录写入 `extraction_mode: "task_anchor_mean"`，便于检测混用。
  - **`outputs/drift_activations/{mistral_7b_v03,gemma2_9b}/` 旧 shard 全部失效，必须 `--no-resume` 重抽**；旧 detector pkl 也需重训。
- **counterfactual pair 数据构造**：新增 `secmcp.data.counterfactual` 把 InjecAgent + ASB 的 attacker instruction pool 注入 AgentTraj-L benign trajectory 的某条 tool response，得到 (clean, attacked) 同 task 同 history pair；两半共享 `metadata.split_group` 强制走同一 split 边。`scripts/01_unify_datasets.py` 增加 `--counterfactual-pairs` 与 `--counterfactual-attacks-per-benign N`。`make_splits` 的非 AgentDojo 分支换成 group-aware `stratified_grouped_split`，pair 不会被打散；旧的 `stratified_split` 行为保留作 backward compat。注入后的 attacked trajectory 在 metadata 直接写明 `malicious_tool_message_indices`，所以弱标签 + 前向传播逻辑能精确定位 step。
- **per-trajectory self-baseline 特征**：`drift_feature_matrix` 新加 `include_self_baseline=True` 与 `prior_state` 参数。对每个 step 用本 trajectory 之前 step 的 `inc_norm / glob_norm / hist_norm / relative_norm` 计算 mean/std 并 z-score；trajectory 第一步无 prior → 输出零向量。这缓解 source/suite 异质性引起的 OOD-benign 假报警。Runtime detector (`SecMCPTaskDriftDetector`) 通过 `extra_args["secmcp_task_drift_prior_state"]` 持久化 `TrajectoryPriorNorms`，逐步累积。`configs/training.yaml` 增加 `task_drift.include_self_baseline: true`。
- **trajectory-level aggregation**：`aggregate_step_scores` 除原有 `max / top2_mean` 外新增 `mean / first_exceed_K / cusum[_wN_sX_rY]`。`first_exceed_K` 把检测预算锁定在前 K 步（一旦攻击没在早期出现就不会再触发，避免 `max` 在长 benign 轨迹上 1−(1−p)ⁿ 的 FPR 爆炸）；`cusum` 是 Page CUSUM，对持续抬升敏感、对孤立 spike 衰减。Diagnostics 新增 `val_trajectory_abort_rate / test_trajectory_abort_rate` 直接出 trajectory FPR。

### 重跑命令（建议顺序）

```bash
# 1. 重新生成 splits（带 counterfactual pair；如不需可去掉 flag）
conda run -n taskdrift python scripts/01_unify_datasets.py \
  --counterfactual-pairs --counterfactual-attacks-per-benign 1

# 2. 重抽 task-anchored 激活（旧 shard 必须丢弃）
for split in train val test; do
  conda run -n taskdrift python scripts/02_extract_activations.py \
    --model mistral_7b_v03 --split $split --mode task_drift \
    --shard-size 100 --log-every 1 --no-resume
done

# 3. 重训 task-drift detector
conda run -n taskdrift python scripts/03_train_detector.py \
  --model mistral_7b_v03 --detector task_drift
```

如要做 aggregation sweep，可临时改 `configs/training.yaml` 的 `task_drift.aggregation`（如 `first_exceed_3` 或 `cusum_w5`）后重训第三步即可，不需要重抽激活。

### Mini 实验集（不重抽激活，分布匹配）

为加快方法学迭代，新增 `scripts/make_mini_subset.py`：从已有 `outputs/splits/*.jsonl` 与 `outputs/drift_activations/{model}/{split}/` 直接切出一个 ~20% 的子集，**不做任何 forward pass**，原产物保持不动。

- 分层：按 `(source, label)` 在 *样本数* 维度上抽 20%（不是按组数，避免 AgentDojo 大组把比例顶到 35-45%）。
- 原子性：仅锁 counterfactual `(clean, attacked)` pair（识别 `metadata.pair_id` 或 `split_group` 以 `cf:` 起头）；AgentDojo `split_group` 在同一 split 内允许内部抽样，因为它本来只是跨 split 防泄漏标记。
- meta：mini shard 的 `sample_index` 重新编号 0..N-1 与新 jsonl 行号对齐，原编号留在 `original_sample_index`。trajectory 聚合按 `sample_index` 分组的逻辑无需改动。
- 输出：`outputs/splits_mini/{train,val,test}.jsonl` + `outputs/drift_activations_mini/{model}/{split}/`，与全量产物并存。

当前 mini 集（fraction=0.20, seed=42）：train 4,907 样本 / 16,938 step；val 1,227 / 4,259；test 1,284 / 4,991。每个 (source, label) bucket 命中 20% ±1%，仅 asb/injecagent label=0（总数 3-15）有 ±5% 抖动。

```bash
# 生成 mini 集（默认覆盖原 mini 目录）
conda run -n taskdrift python scripts/make_mini_subset.py --fraction 0.20 --seed 42

# 在 mini 集上训 detector（其它 flag 同全量）
conda run -n taskdrift python scripts/03_train_detector.py \
  --model mistral_7b_v03 --detector task_drift \
  --drift-activation-root outputs/drift_activations_mini

# 给新模型抽 mini 激活：让 02 脚本读 splits_mini/，写 drift_activations_mini/
conda run -n taskdrift python scripts/02_extract_activations.py \
  --model gemma2_9b --split train --mode task_drift \
  --splits-dir outputs/splits_mini --output-root outputs/drift_activations_mini --no-resume

# 仅生成 splits_mini jsonl，跳过激活复制
conda run -n taskdrift python scripts/make_mini_subset.py --no-activations
```

  你现在应该跑这个，用 mini split 重新抽较小激活集，然后训练 mini detector：

  for split in train val test; do
    conda run -n taskdrift python scripts/02_extract_activations.py \
      --model mistral_7b_v03 \
      --split $split \
      --mode task_drift \
      --splits-dir outputs/splits_mini \
      --output-root outputs/drift_activations_mini \
      --shard-size 100 \
      --log-every 1 \
      --no-resume
  done

  然后训练：

  conda run -n taskdrift python scripts/03_train_detector.py \
    --model mistral_7b_v03 \
    --detector task_drift \
    --drift-activation-root outputs/drift_activations_mini \
    --output-root outputs/detectors_mini



- Hard-negative loop：从 `diagnostics.test_group_diagnostics` 和 false-positive benign trajectories 中定位误报最多的 suite/tool/sample_type，把这些 benign tool steps 加入更高权重的校准集或训练权重。
- Step-level label 进一步改进：提高 AgentDojo injection 片段匹配覆盖率；匹配不到的 positive trajectory 后续可改为 multiple-instance learning。
- Source/suite-normalized anchors：为 AgentDojo suite 或 tool type 维护 benign incremental/global anchor，而不是所有 benign step 共用一组 anchor。
- 系统化 sweep：`aggregation=max/top2_mean`、`n_anchors=500/1000/2000`、`include_anchor_distances=true/false`、`layers.mode=concat/last_only`、不同 layer 组合，统一写入 summary 表。
- Runtime benchmark：实现 `scripts/04_eval_agentdojo.py` 后，以 utility/security、ASR、benign false abort rate、trigger turn、trigger tool 为主指标；离线 accuracy 只作为辅助诊断。

## Phase 5 落地记录

### 已做

- 在 `taskdrift` env 安装 AgentDojo：`pip install -e data/AgentDojo --no-deps`，并补齐 `anthropic / cohere / google-genai / openai / docstring-parser / tenacity / python-dotenv / deepdiff / email-validator / pydantic[email]`。
- `src/secmcp/integrations/agentdojo_drift_detector.py`：detector 在每个 tool step 通过 `Logger.set_contextarg` 把 `secmcp_task_drift_{scores,trajectory_score,aggregation,threshold,triggered,trigger_step}` 写进 AgentDojo TraceLogger，跟随 per-task JSON 持久化。
- `scripts/04_eval_agentdojo.py`：CLI 驱动，三种 defense 模式 `none / shadow / abort`，每个 suite 同时跑 clean (`benchmark_suite_without_injections`) 与 attacked (`benchmark_suite_with_injections`)；采样比例默认读 `configs/eval.yaml`，可用 `--user-task-frac / --injection-task-frac / --max-user-tasks / --max-injection-tasks` 覆盖；pipeline 手工构造（不走 `AgentPipeline.from_config`，因为它 `defense` enum 不含 secmcp）；`--dry-run` 不调 LLM 即可列出抽样 task。
- `src/secmcp/integrations/agentdojo_eval.py`：扫 `{logdir}/{pipeline_name}/{suite}/.../*.json`，输出 `metrics.json`（per-suite + overall：utility_clean / utility_attacked / security_attacked / ASR / detector_trigger_rate / benign_false_abort_rate / mean_trigger_step）和 `step_scores.jsonl`（每条 trajectory 的 per-step score 与 trigger 信息）。
- 测试：`tests/test_agentdojo_eval.py`（聚合器 2 用例），原有 `tests/test_agentdojo_drift_detector.py` 仍通过。

### 输出目录

```text
outputs/eval/agentdojo/
├── runs/                                   # AgentDojo 原生 per-task trace JSON
│   └── {llm}-{defense_tag}/{suite}/{user_task}/{attack}/{injection}.json
└── {llm}-{defense_tag}/
    ├── run_summary.json                    # 本次抽样的 suite/user/injection 列表
    ├── metrics.json                        # 聚合指标
    └── step_scores.jsonl                   # 每条轨迹的 SecMCP per-step 分数
```

其中 `defense_tag ∈ {no_defense, secmcp_shadow_{model}, secmcp_abort_{model}}`。

### 接下来要跑的命令

前置：`export OPENAI_API_KEY=...`；detector pkl 默认查 `outputs/detectors/{model}/task_drift_best.pkl`，当前只有 mini 版，所以先用 `--detector-path` 指过去。

```bash
# 1. dry-run 验抽样（不调 LLM、不加载模型）
conda run -n taskdrift python scripts/04_eval_agentdojo.py \
  --suite workspace --max-user-tasks 1 --max-injection-tasks 1 --dry-run

# 2. 冒烟：workspace 1×1，三档 defense 全跑
for D in none shadow abort; do
  conda run -n taskdrift python scripts/04_eval_agentdojo.py \
    --detector-path outputs/detectors_mini/mistral_7b_v03/task_drift_best.pkl \
    --suite workspace --max-user-tasks 1 --max-injection-tasks 1 \
    --defense $D
done

# 3. 正式（用 eval.yaml 的 20% 抽样，4 个 suite，important_instructions 攻击）
#    顺序：no_defense baseline → secmcp_shadow 校准阈值 → secmcp_abort 出主结果
conda run -n taskdrift python scripts/04_eval_agentdojo.py --defense none
conda run -n taskdrift python scripts/04_eval_agentdojo.py --defense shadow \
  --detector-path outputs/detectors/mistral_7b_v03/task_drift_best.pkl
conda run -n taskdrift python scripts/04_eval_agentdojo.py --defense abort \
  --detector-path outputs/detectors/mistral_7b_v03/task_drift_best.pkl \
  --threshold <从 shadow 的 step_scores.jsonl 校出来的值>

# 4. 不用 OpenAI：先 `vllm serve /hub/huggingface/models/MistralAI/Mistral-7B-Instruct-v0.3`
#    再加 --llm-provider local --llm-model mistral
```

常用 flag：`--suite workspace slack banking travel`、`--attack important_instructions`、`--skip-clean / --skip-attacked` 拆开跑、`--force-rerun` 覆盖已存在的 trace。

### 已知遗留

- 我没有 `OPENAI_API_KEY`，所以**真正的 end-to-end smoke run 没跑成**；上面 dry-run + pipeline 构造已验证，剩下需要你配 key 实跑一次。
- 默认 detector 路径指向 `outputs/detectors/mistral_7b_v03/task_drift_best.pkl`，但当前仓库里只有 `rf_anchor_best.pkl` 和 `outputs/detectors_mini/.../task_drift_best.pkl`；想用默认路径请先完成 PLAN 上面那段重跑流程，或始终通过 `--detector-path` 指过去。

## 2026-05-15 运行记录

- ChatAnywhere key 不能直连 OpenAI 官方端点；需设 `OPENAI_BASE_URL=https://api.chatanywhere.tech/v1`，模型名用 `gpt-4o-mini-2024-07-18` 以匹配 AgentDojo attack 模板。
- AgentDojo eval 脚本已修：ASR 不再反向计算；false abort 改用 `secmcp_task_drift_aborted`；regular clean 与 injection-task-as-user 分开；pipeline 名加入 threshold / sample cap / seed，避免旧 trace 混入。
- mini Mistral detector 在 workspace 8 user × 3 injection 的高阈值 `0.98964` 下：shadow ASR `0.125`，abort ASR `0.0417`，regular-clean false abort `0`；结论只作 smoke，不作正式结果。
- `none` baseline 曾因模型返回非法 tool-call JSON 崩溃；这是 ChatAnywhere/模型输出稳定性问题，不是 detector 问题。
- 当前 `outputs/splits` 是 5 月 14 日重生成过的 step-locator 版本，但未发现 counterfactual pair 标记；若要严格按最新 counterfactual 方案，需重建 splits 并重抽激活。

## 下一步

- 继续抽全量 Mistral/Gemma task-drift activations；中断可 resume，split 改动后才用 `--no-resume` 重抽。
- 训练正式 detector 后再跑 AgentDojo；mini detector 只用于 pipeline smoke。
- AgentDojo 多样本评估要显式加 `--user-task-frac 1.0 --injection-task-frac 1.0`，否则先按 `eval.yaml` 的 20% 采样再 cap。
- baseline / shadow / abort 分开跑；若 `none` 继续 JSONDecodeError，换官方 OpenAI key 或更稳 endpoint。

## 2026-05-25 Qwen3-32B 激活提取记录

- 原 `device_map:auto` 四卡只做层切分，GPU 利用率低；`--batch-steps 2` 对 HF/Accelerate 路径帮助有限。
- 已新增 opt-in 参数：`--batch-steps`（默认 `1`）和 `--tensor-parallel`（默认关闭）。不传新参数时旧单卡/旧多卡 `device_map` 路径不变。
- 新建独立环境 `taskdrift-tp` 用于 TP，未升级原 `taskdrift` 环境；当前关键版本：`torch 2.5.1+cu121`、`transformers 4.57.6`、`accelerate 1.13.0`。
- 已卸载 `taskdrift-tp` 中不匹配且本任务不用的 `torchvision/torchaudio`，避免 Transformers 导入 Qwen3 时触发 `torchvision::nms` 错误。
- Qwen3-32B train 已写出若干完整 shard 到 `outputs/drift_activations/qwen3_32b/train/`，脚本会自动 resume。

常用检查：

```bash
find outputs/drift_activations/qwen3_32b/train -maxdepth 1 -name 'drift_*.pt' | wc -l
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

继续跑 Qwen3-32B train（TP，0-3 卡，较稳版本）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_ADDR=127.0.0.1 MASTER_PORT=29545 \
MPLCONFIGDIR=/tmp/mpl-secmcp-qwen PYTHONPATH=src \
/data/home/Haoran/miniconda3/envs/taskdrift-tp/bin/torchrun \
  --nnodes 1 --nproc-per-node 4 \
  --master-addr 127.0.0.1 --master-port 29545 \
  scripts/02_extract_activations.py \
  --model qwen3_32b --split train --mode task_drift \
  --shard-size 100 --log-every 10 \
  --batch-steps 1 --tensor-parallel
```

同样方式跑 val/test：只改 `--split val` 或 `--split test`，建议换一个未占用的 `MASTER_PORT`。
