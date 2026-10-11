# Sequence-aware Jev Reranking：序列推荐实验方案

> 文档状态：**实验设计与执行清单**，不是实验结果。所有结果位均保留为 `TBD`，不得把计划性判断写成已验证结论。
>
> 目标：验证将 Jev 作为决策型重排器接在序列推荐召回/排序之后，能否在长期偏好与近期会话意图冲突时改善下一物品推荐，并判断收益来自何种状态表示。

## 1. 研究范围、先验与可验证主张

### 1.1 名称与组合方式

- **Jev**：按 choice 概率输出结构化决策的 hosted 模型。这里采用“候选集重排”用法：传统序列模型先召回 Top-N，Jev 根据结构化用户状态和候选元数据输出各候选的选择概率。
- **IDNP**：`IDNP: Interest Dynamics Modeling Using Generative Neural Processes for Sequential Recommendation`，核心是多尺度短期兴趣与动态长期兴趣建模（Du et al., 2022）。本方案把 IDNP 作为序列状态编码器/基线；“IDNP + Jev”是待验证组合，不把它写成已有工作结论。
- 直接先验：Jev 推荐重排实证研究使用最近行为作为 state、候选项目作为 choice，并在 Amazon Reviews 2023 上以候选规模 20/50/100/200 做 hard-candidate reranking；该协议可复用，但本方案增加长期/短期解耦、时间间隔、冲突子集和本地蒸馏。

### 1.2 研究问题与假设

| RQ | 研究问题 | 可证伪假设 | 首要证据 |
|---|---|---|---|
| RQ1 | Jev 重排是否改善序列推荐，尤其是长期/短期冲突？ | H1：在冲突子集上，`Base + Jev` 的 NDCG@10/Hit@10 高于同一 Base 的原始排序；总体收益不应作为唯一结论。 | 冲突子集配对差值、95% CI |
| RQ2 | 双时间尺度状态是否优于单一历史？ | H2：`long + recent` 优于 `recent-only` 和 `long-only`；冲突强度越高，差值越大。 | 分层 NDCG/intent-match |
| RQ3 | 时间间隔和行为类型是否提供额外信息？ | H3：加入 gap 与 behavior type 后，Jev 的选择概率校准和冲突识别提升。 | ECE/Brier、冲突准确率 |
| RQ4 | 直接拼接还是解耦后融合更有效？ | H4：独立 long/session 编码并用 gated fusion 的结果优于简单 concat；若失败，应报告冲突是状态构造还是 Jev 决策问题。 | 组件消融、交互效应 |
| RQ5 | Jev 的置信度能否安全控制重排？ | H5：低置信度 fallback（回退到 Base）降低极端错误率，并保持主要指标不显著下降。 | selective risk、coverage-risk 曲线 |
| RQ6 | 哪些稀疏场景受益？ | H6：文本丰富的长尾 item 和多意图用户可能受益；新用户仅靠 Jev 不能弥补召回缺失。 | 用户/物品冷启动分层、召回覆盖率 |
| RQ7 | 能否把 hosted Jev 的决策蒸馏到本地？ | H7：蒸馏学生保留部分冲突场景收益，同时降低延迟和调用成本。 | NDCG、agreement、p95、cost |

### 1.3 机制图

```text
历史行为 (item, timestamp, behavior, metadata)
        ├── 长期兴趣编码器 / IDNP long-term state
        └── 近期会话编码器 / IDNP short-term state
                      ↓
        结构化状态：long、session、gap、behavior、uncertainty
                      ↓
        SASRec/BERT4Rec/IDNP 召回 Top-N hard candidates
                      ↓
        Jev choice 概率 + Base score 融合/门控
                      ↓
        top-K 推荐（可选多样性和长尾约束）
```

Jev 不替代长期兴趣动态学习；长期状态由 IDNP 或本地序列编码器生成，Jev 负责在候选间做上下文决策。文本化序列必须保留顺序、行为类型和时间间隔，不能只拼接 item 标题。

## 2. 数据集、任务与预处理

### 2.1 数据集优先级

**主数据集：Amazon Reviews'23**（McAuley Lab，优先选 `Books`、`Movies & TV`、`Video Games`）。该数据同时提供 review、timestamp、item metadata/title/category，适合构造时间序列、文本语义和长尾切片。来源：[Amazon Reviews'23](https://amazon-reviews-2023.github.io/main.html)。

**复现实验：MovieLens-1M 或 MovieLens-20M**。MovieLens-1M 含约 1M ratings、约 6k users 和 4k movies，数据规模适合先跑完整矩阵。来源：[GroupLens MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)。

**可选第三数据集：RetailRocket 或 Yoochoose**。只有在存在可靠 timestamp/session/行为字段且下载许可明确时加入；否则不把它作为主结论依据。

### 2.2 交互定义

1. 按 `(user, timestamp)` 排序；同一用户同一物品的重复记录按时间保留，敏感性实验再去重。
2. Amazon 默认将 review/rating 视为正反馈；MovieLens 使用评分阈值（主协议 `rating >= 4`，敏感性 `>=3`）。如果同时保留 rating，rating 仅作为历史行为类型或辅助特征，不能泄漏未来评分。
3. session 切分：相邻交互间隔 `gap >= 30 min` 产生新 session；做 `15/60 min` 敏感性。每个 session 至少保留 2 个交互用于冲突分析。
4. 过滤规则在 train 之前确定并记录：主协议 `min_user_interactions=10`、`min_item_interactions=5`；另保留未过滤副本用于冷启动分析（不把过滤后的结果称为完整冷启动）。
5. item 文本只使用时间截点以前可获得的 title/category/description。清理 HTML、控制长度（title 64 tokens，description 256 tokens）；保留原始 item ID 与脱敏映射。

### 2.3 时间切分与任务

- **主任务**：next-item prediction。对每个用户按时间排序，倒数第 2 个交互做 validation，最后 1 个做 test；训练样本用前缀到每个时间点的 next-item 对。
- **稳健性切分**：全局时间 80/10/10；再报告跨时间漂移（早期训练、后期测试）。
- **候选协议**：Base 模型先取 Top-200；主报告 Top-20/50/100/200。Jev 只在候选集内重排，必须记录候选召回率，避免把 reranker 的收益误解为召回收益。
- **负样本**：离线全量 ranking 使用可索引 item 集；若成本过高，固定每个正例的 100 个 negatives，并报告 sampled/full protocol，不混在一张表中。
- **数据泄漏检查**：所有用户画像、类别统计、item popularity、文本摘要和候选负样本只能由 train/当前 prefix 生成；测试时间之后新增的 metadata 不可使用。随机打乱时间的结果只作反例诊断，不作主结果。

### 2.4 冲突子集（主验证集）

冲突样本必须在运行前固定规则、阈值和样本 ID，保存到 `data/conflict_manifest.parquet`。

对测试时刻 `t` 的用户定义：

- **长期分布** `p_L`：`t` 前至少 30 天或最近 180 天内的历史交互按类别/文本主题聚合，使用可复现的指数衰减 `w=exp(-Δt/τ_L)`，主协议 `τ_L=90d`。
- **近期分布** `p_S`：当前 session 最近 `K=5` 条行为（不足 5 条则用实际长度）；另做 `K=3/10` 敏感性。
- **冲突强度**：`JSD(p_L || p_S)`，并要求 `argmax(p_L) != argmax(p_S)`、两侧各至少 2 条行为、当前候选中同时存在长期匹配项与近期匹配项。
- **强冲突**：`JSD >= 0.5`（阈值在 validation 固定）；**中冲突**：`0.2 <= JSD < 0.5`；其余为 non-conflict。若类别缺失，使用冻结的 item text embedding 聚类；聚类模型只能在 train 拟合。
- **候选配对**：从同一 hard-candidate pool 中取 `c_L=argmax sim(item,p_L)` 与 `c_S=argmax sim(item,p_S)`；记录二者是否同时存在及真实 next item 属于哪一侧。只有两者均在候选池中的样本用于“选择偏好”指标；总体 test 仍保留。
- **切换标记**：若 session 起点与长期 top 类别不同且起点前 gap >=30 min，标记 `switch=1`；报告 switch 与 conflict 分开，避免把时间间隔当成冲突本身。

## 3. 方法、基线与公平协议

### 3.1 必要基线

| 类别 | 方法 | 作用与要求 |
|---|---|---|
| 传统序列 | GRU4Rec 或 NARM、SASRec、BERT4Rec | 递归/因果 Transformer/双向 Cloze 参照；固定同一 train/val/test 和候选池。SASRec：[论文](https://arxiv.org/abs/1808.09781)；BERT4Rec：[论文](https://arxiv.org/abs/1904.06690)。 |
| 兴趣动态 | IDNP | 最近的 long/short dynamics 参照；优先使用作者代码与原始超参，无法复现时标记为 unresolved，不用自写简化版替代主基线。IDNP：[论文](https://arxiv.org/abs/2208.04600)。 |
| 排序器 | Base pointwise/listwise ranker（DCNv2 或同等 MLP） | 检验收益是否只是更强的二阶段排序。 |
| 决策模型 | Jev-only | 只给结构化状态和候选，不输入 Base score，隔离 Jev 自身决策作用。 |
| 语言模型参照 | Qwen pointwise/listwise（可选） | 与 Jev 的 hosted decision 输出作成本/效果对比；记录模型版本和提示模板。 |
| 简单上界/下界 | Base 原序、按 popularity、随机候选顺序 | 检查数据和候选管线，不能替代强基线。 |

基线实施矩阵需保存 `source/revision/config/dataset/split/checkpoint/status`。只有完整可核验版本进入“主结果”，简化或 debug 版本只能进入开发日志或消融。

### 3.2 待验证方法变体

| ID | 变体 | 长期 | 近期 | gap/type | 融合与 Jev |
|---|---|---|---|---|---|
| A | `S-only` | — | session encoder | 否 | Base / Jev |
| B | `L-only` | long encoder | — | 否 | Base / Jev |
| C | `L+S-concat` | encoder | encoder | 可选 | 直接 concat + MLP |
| D | `L/S-gated` | 独立 encoder | 独立 encoder | 否 | learned gate `g` |
| E | `D+gap` | 独立 | 独立 | time gap | gap bucket/continuous |
| F | `D+type` | 独立 | 独立 | behavior type | embedding |
| G | `D+gap+type` | 独立 | 独立 | 两者 | full state |
| H | `IDNP+Jev` | IDNP long function | IDNP multi-scale short | 两者 | Jev choice reranking |
| I | `Jev-gated` | G/H state | G/H state | 两者 | 置信度门控/fallback |
| J | `Jev-distill` | G/H teacher | local student | 两者 | KL/CE 蒸馏 |

主方法暂用名：`Sequence-aware Jev Reranking (SA-Jev)`。正式论文名称在完成可复现性核验后再定。

### 3.3 Jev 输入、分数与门控

每次调用保存不可变 JSON（不保存个人可识别信息）：

```json
{
  "state": {
    "long_term": [{"category": "...", "weight": 0.0}],
    "recent_session": [{"item_id": "...", "category": "...", "behavior": "..."}],
    "time_gaps_min": [0, 30],
    "uncertainty": 0.0
  },
  "choices": [{"candidate_id": "...", "title": "...", "category": "...", "base_rank": 1}],
  "task": "choose the item most likely to be the next positive interaction"
}
```

- 只把候选作为 choice，不让 Jev 生成候选；候选 ID、顺序、标题脱敏映射均记录。
- 概率融合：`s = α·z(s_base) + (1-α)·z(log(p_jev+ε))`；`α` 在 validation 上网格搜索 `{0, .25, .5, .75, 1}`，不在 test 调参。
- 置信度 `conf = p_top - p_second`，辅助指标使用 entropy；若 `conf < τ`、API 超时或 schema 校验失败，回退 Base。`τ` 在 validation 固定，报告 coverage-risk 曲线。
- 每次调用记录 Jev 版本、提示模板哈希、候选顺序、响应 hash、延迟、重试次数、失败率、token/cost（如 API 提供）。Jev 结果需缓存，保证重复评估可复现并控制成本。
- 为检查选项敏感性，validation/test 各抽取固定样本做候选顺序随机化、ID 标签重命名和 prompt 模板变体；只把通过鲁棒性协议的版本用于主叙述。

## 4. 评价指标与切片

### 4.1 主要准确性指标

- `Recall@{1,5,10,20}`、`Hit@{1,5,10}`、`NDCG@{5,10,20}`、`MRR`；主指标预注册为 **冲突子集 NDCG@10**，次指标为总体 NDCG@10。
- 候选召回率 `Recall_candidate@N` 单独报告；若正例未进候选池，Jev 不计为排序失败。
- Jev 概率：candidate-level log loss、Brier、ECE（等宽/等频 bins 均报告 bin 数）。

### 4.2 多样性、长尾与业务约束

- `CatalogCoverage@K`、`ILD@K`（item embedding cosine distance）、intra-list category entropy。
- `Novelty@K`（按 train popularity 的 `-log p(item)`）、长尾曝光比例（底部 20%/50% item popularity）、Gini exposure。
- 若业务有约束，报告重复 item/category、价格/库存/安全过滤通过率；约束后再比较准确性，不能只报告未约束列表。

### 4.3 冲突、意图与时间切换

- `NDCG@10_conflict`、`NDCG@10_nonconflict`、按 JSD 强度分桶的曲线。
- `intent-match@K`：推荐列表与真实 next item 所属近期/长期主题的匹配；分别报告 `short-match`、`long-match`，不把主题匹配当成点击真实性。
- `short-vs-long win rate`：在候选同时包含 `c_S,c_L` 的样本上，Jev top-1 选择近期匹配项、长期匹配项及真实标签的比例。
- `switch recovery`：检测到 session switch 后的前 K 个位置 NDCG/Hit，以及 gap 分桶结果。

### 4.4 冷启动、稀疏与长尾

- 用户按 train 行为数：`1–3`、`4–9`、`10–30`、`>30`；若主过滤排除前两组，则使用未过滤副本单独评估。
- item 按 train 频次：`1`、`2–5`、`6–20`、`>20`；另报告“新 item 在候选池中”的条件结果。
- 序列长度、session 长度、主题熵、冲突强度、候选召回率五维分层。每个切片报告样本数，少于预设最小样本量（建议 200 users）只作为描述性结果。

### 4.5 效率与成本

报告 offline training GPU-hours、显存峰值、每用户候选数、Jev calls/request、cache hit rate、p50/p95 API latency、timeout/error rate、每 1k requests cost。Hosted Jev 与本地模型硬件不同，不能把未经归一化的绝对延迟直接声称为模型优越。

## 5. 消融、鲁棒性与统计检验

### 5.1 消融逻辑

| 消融 | 移除/替换 | 验证机制 | 结果位 |
|---|---|---|---|
| `w/o long` | 去掉长期状态 | 长期偏好是否有独立贡献 | TBD |
| `w/o recent` | 去掉当前 session | 短期意图是否必要 | TBD |
| `concat → gate` | 简单拼接替换为 gated | 解耦融合是否有效 | TBD |
| `w/o gap` | 删除时间间隔 | 时间连续性是否提供信息 | TBD |
| `w/o type` | 删除行为类型 | 不同行为信号是否有贡献 | TBD |
| `Jev score → random` | 随机重排候选 | Jev 决策是否真正贡献 | TBD |
| `Jev → base` | 全量回退 Base | 门控的必要性 | TBD |
| `no text` | 删除标题/类别文本 | Jev 语义理解是否依赖 item metadata | TBD |
| `IDNP state → concat` | 用简单拼接替换 IDNP | IDNP 动态状态是否贡献 | TBD |
| `teacher → student` | Jev 换本地学生 | 蒸馏收益与成本权衡 | TBD |

每个消融只改变一个机制；消融不可替代完整主方法比较。

### 5.2 鲁棒性矩阵

1. 候选规模 `N={20,50,100,200}`。
2. session gap `{15,30,60}` min，短期窗口 `K={3,5,10}`，长期衰减 `τ={30,90,180}` days。
3. 候选顺序随机化、候选 ID 重命名、标题截断/同义改写、类别字段缺失 10%/30%。
4. API 失败/超时模拟 `{0,1,5}%`，验证 fallback 和重试上限；不把重试后的成功调用当作零成本。
5. seed 至少 3 个，主结果建议 5 个；数据切分固定，模型初始化、负采样和候选顺序使用独立 seed。
6. 在同一用户上做 paired bootstrap（1000 次）得到 95% CI；主比较使用 Wilcoxon signed-rank 或 paired permutation test，报告 effect size（Cliff's delta 或 paired Cohen's d）。多指标/多切片用 Holm correction，预先指定 primary metric，避免事后挑选。
7. 置信度校准：validation 拟合温度/等距校准只能用于验证集；test 只报告校准后结果，不重新拟合。

## 6. 案例分析与失败模式

### 6.1 采样与展示

从 test 固定抽取：10 个强冲突成功案例、10 个强冲突失败案例、5 个短序列、5 个长尾 item、5 个 Jev 低置信度 fallback；同一 case 同时展示 Base、Jev、full SA-Jev 的候选分数、top-K、长期/近期摘要、gap/type 和真实 next item。案例表字段：`user_hash, cutoff_time, conflict_strength, candidate_pair, base_rank, jev_prob, final_rank, label, failure_tag`。

### 6.2 失败标签

- `candidate_missing`：真实 item 不在召回池；归因于召回，不归因于 Jev。
- `stale_long_state`：长期摘要压过明确的新 session。
- `overreact_short_state`：短期噪声或单次点击被过度放大。
- `metadata_ambiguity`：标题/类别无法区分语义。
- `gap_misread`：间隔不能代表兴趣切换。
- `popularity_bias`：Jev 偏向热门 item，长尾曝光下降。
- `order_or_label_sensitivity`：候选顺序/ID 命名改变选择。
- `api_or_schema_failure`：超时、拒答、格式错误导致 fallback。

案例解释仅作为诊断证据，不把 Jev 的自然语言理由当作因果解释；主证据仍是配对指标和失败率。

## 7. 统计表、图与结果占位

### 表 1：数据与冲突切片统计

| Dataset | #users | #items | #interactions | #sessions | conflict % | strong-conflict % | TBD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Amazon Books | TBD | TBD | TBD | TBD | TBD | TBD | |
| Amazon Movies & TV | TBD | TBD | TBD | TBD | TBD | TBD | |
| Amazon Video Games | TBD | TBD | TBD | TBD | TBD | TBD | |
| MovieLens-1M | TBD | TBD | TBD | TBD | TBD | TBD | |

### 表 2：主结果（每个方法均为相同候选池）

| Method | Dataset | Recall@10 | NDCG@10 | NDCG@10-conflict | Hit@10-conflict | CatalogCoverage | Long-tail exposure | p95 latency | Jev cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SASRec | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | — |
| BERT4Rec | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | — |
| IDNP | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | — |
| IDNP + Jev | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SA-Jev (full) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 表 3：组件消融

| Variant | long | recent | gap | type | fusion | Jev | NDCG@10-conflict | ECE | TBD |
|---|---|---|---|---|---|---:|---:|---:|---:|
| full | ✓ | ✓ | ✓ | ✓ | gated | ✓ | TBD | TBD | |
| w/o long |  | ✓ | ✓ | ✓ | gated | ✓ | TBD | TBD | |
| w/o recent | ✓ |  | ✓ | ✓ | gated | ✓ | TBD | TBD | |
| concat | ✓ | ✓ | ✓ | ✓ | concat | ✓ | TBD | TBD | |
| w/o gap/type | ✓ | ✓ |  |  | gated | ✓ | TBD | TBD | |
| no Jev | ✓ | ✓ | ✓ | ✓ | gated |  | TBD | TBD | |

### 图计划

1. `Fig.1` 方法与数据流图（状态构造 → recall → choice rerank → fallback）。
2. `Fig.2` 按冲突强度的 NDCG@10 差值曲线。
3. `Fig.3` coverage-risk 曲线：Jev confidence threshold 与错误率/覆盖率。
4. `Fig.4` 长尾曝光与准确性 Pareto 曲线。
5. `Fig.5` 候选规模、prompt/label 扰动的性能箱线图。

## 8. 执行顺序与 A5000 资源计划

### P0：数据与管线验证（低成本）

- 下载 MovieLens-1M；运行统计、时间切分、候选构造、冲突 manifest、指标单元测试。
- 在 1 个 seed、1 个数据集、N=20 上跑 SASRec smoke；smoke 只验证数据流，不进入论文表。

### P1：本地强基线（中成本）

- MovieLens-1M：GRU4Rec/NARM、SASRec、BERT4Rec、IDNP（优先官方实现）。
- Amazon 三域选一做完整 train/valid/test；统一候选池和 3/5 seeds。

### P2：核心机制（中到高成本）

- `S-only/L-only/concat/gated/gap/type/full`；先在 MovieLens + Amazon Books 验证。
- 固化阈值、冲突 manifest、主要指标后再扩展域。

### P3：Jev 调用（API 成本/网络依赖）

- 先在 validation 预计算 Jev response cache；只在 test 使用冻结 cache。
- 运行 Jev-only、Base+Jev、confidence gating、候选 N 规模实验；记录版本、成本、延迟、失败率。
- 需要用户提供可用 Jev API key/组织额度；没有 key 时先完成本地部分，Jev 单元保持 `blocked-by-credential`，不伪造结果。

### P4：稳健性与蒸馏（高成本，可放附录）

- label/order/prompt 扰动、gap/K/τ 敏感性、API dropout、student distillation。
- 仅在 P3 的主比较有稳定结果后执行。

### 资源约束

- A5000 24 GiB 足够运行 SASRec/BERT4Rec/IDNP 的中等规模配置和本地蒸馏；batch、max sequence length、mixed precision 应根据显存逐步调节。
- 当前 GPU 监测到约 18.7/24.6 GiB 已被其他进程占用，训练前必须等待空闲或申请独占；禁止在共享 GPU 上强行杀进程。
- `/media/sdust307/disk3` 实际位于根分区，剩余约 40 GiB 且使用率 96%；raw Amazon 数据和 checkpoints 优先放 `/media/sdust307/disk2`（剩余约 167 GiB），实验配置/日志保留在本目录。
- Jev 是 hosted API，延迟不能与本地 GPU 直接比较；离线缓存、调用预算和网络失败率必须进入实验日志。

## 9. 预期结论与 IDNP × Jev 验证路径

这里只定义**判定规则**，不预填方向或数值：

1. 若 `IDNP + Jev` 相比 `IDNP` 在 conflict 与 strong-conflict 的 paired NDCG@10 置信区间全为正、经过 Holm 校正仍显著，且 non-conflict 不出现预设阈值（建议 1% 绝对 NDCG）以上下降，则支持“Jev 能利用 IDNP 的 long/short 状态做冲突重排”。
2. 若只有带 `gap/type` 的状态有收益，支持时间/行为结构是必要信息；若只在有文本 metadata 时有收益，应把结论收窄为“语义丰富候选的重排收益”。
3. 若 `concat` 与 `gated` 无差异，不能声称解耦机制必要；可保留 Jev 重排本身作为主要结果。
4. 若总体指标不变但 conflict 指标提升，同时长尾/多样性不恶化，说明价值集中在困难子集；论文主张应围绕该子集，不报告泛化为所有推荐场景。
5. 若收益来自更高候选召回而非重排，必须把贡献归因于 recall pipeline，并重新设计 Jev 实验。
6. 若 Jev 选择对 order/label 显著敏感、ECE 较差或 fallback 率高，则结论应限定为“需鲁棒提示/校准的 hosted reranker”，并把风险列为核心限制。
7. 若学生模型在相同候选池上保留冲突收益且显著降低 p95/cost，可支持部署路径；否则 Jev 仅作为离线分析工具。

## 10. 证据与引用

- IDNP：Du et al., *IDNP: Interest Dynamics Modeling Using Generative Neural Processes for Sequential Recommendation*, [arXiv:2208.04600](https://arxiv.org/abs/2208.04600)。
- Jev 推荐重排先验：Lyu & Xia, *Decision-Oriented Recommendation Reranking: An Empirical Study of Jev*, [arXiv:2609.40241](https://arxiv.org/abs/2609.40241)。
- SASRec：Kang & McAuley, [arXiv:1808.09781](https://arxiv.org/abs/1808.09781)。
- BERT4Rec：Sun et al., [arXiv:1904.06690](https://arxiv.org/abs/1904.06690)。
- LLM-SRec：*Lost in Sequence: Do Large Language Models Understand Sequential Recommendation?*, [arXiv:2502.13909](https://arxiv.org/abs/2502.13909)。该工作提示仅把行为序列文本化不等于真正利用顺序，因此本方案显式传递顺序、gap 和 behavior。
- Amazon Reviews'23：[官网](https://amazon-reviews-2023.github.io/main.html)。
- MovieLens：[GroupLens 1M](https://grouplens.org/datasets/movielens/1m/)。

---

**No-fabrication status：** 当前没有生成任何实验结果。表格中的 `TBD` 只能由实际运行、论文中协议完全匹配的公开结果或可核验的 baseline 报告填充。
