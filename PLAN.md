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
- trajectory-level score 默认从 step-level score 聚合，优先比较 `max` 与 `top-2 mean`。

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
- Phase 3 的 task-drift 激活提取代码已经实现：每个 `role == "tool"` 的消息会生成 `task_prefix`、`history_prefix`、`post_tool_prefix`，输出 `drift_*.pt`、`labels_*.pt`、`meta_*.jsonl` 到 `outputs/drift_activations/{model}/{split}/`。脚本 `scripts/02_extract_activations.py` 当前默认走 `task_drift` 模式，旧 whole-context 仍可通过 `--mode whole_context` 显式运行。
- Phase 4 的主 detector 已经可训练：`HistGradientBoostingClassifier`、benign incremental/global drift anchors、`include_anchor_distances` 配置、每层 norm/cosine、trajectory-level `max` / `top2_mean` score 聚合、validation threshold 选择、metrics 保存都已经在代码中。旧 `rf_anchor` 和 `logistic_diff` whole-context baseline 训练代码也保留。
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
- `outputs/drift_activations/` 目前为空，说明新 task-drift 主方案还没有可训练的本地激活产物。
- `outputs/detectors/mistral_7b_v03/rf_anchor_best.pkl` 已存在，属于旧 whole-context RF baseline，不是 task-drift detector。

### 最近下一步

当前 split 已经确认能枚举出 tool steps；下一步是按目标模型生成 `train`、`val`、`test` 三个 split 的 task-drift 激活，然后训练 `task_drift` detector。只有这两步完成后，离线 AUROC/AUPRC/FPR@95TPR 才能代表新方案的主线结果。AgentDojo / ASB runtime 评估和 baseline 对比仍需要后续补脚本。
