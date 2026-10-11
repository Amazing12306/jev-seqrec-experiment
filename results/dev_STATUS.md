# Jev × 序列推荐实验状态（更新于 2026-10-11）

## 最新进度：full-SASRec × Jev 正式 test 已完成

### 正式 full-SASRec 结果

- 上游实现：`SASRec.pytorch` pinned revision `b253407b3c5d6ec3201a39767a572f44fd5ef6d0`；`maxlen=200`、hidden size 50、2 blocks、1 head，validation early stopping。
- 三个 seed 均为完整训练配置；每个 test case 有 200 个 SASRec 候选，Jev 只做候选内重排，不改变召回。
- 6 个 response cache 均覆盖 5,779 个 case、成功率 100%；validation seed2028 的一次超时已补跑，合并时去重。

| Seed | Base NDCG@10 | Jev NDCG@10 | Δ | Conflict base → Jev | Strong conflict base → Jev | alpha / threshold / Jev coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 2027 | 0.04273 | 0.04081 | −0.00192 | 0.03976 → 0.03575 | 0.04697 → 0.04622 | 0.75 / none / 100.0% |
| 2028 | 0.04406 | 0.04407 | +0.00001 | 0.03874 → 0.03901 | 0.04619 → 0.04667 | 0.75 / 0.4 / 4.6% |
| 2029 | 0.04655 | 0.04666 | +0.00011 | 0.04360 → 0.04309 | 0.05792 → 0.05642 | 0.25 / 0.4 / 4.2% |

用户级三 seed 配对结果：

| 分组 | Base | Jev | Δ | 95% bootstrap CI | paired permutation p |
|---|---:|---:|---:|---:|---:|
| 全部 | 0.04445 | 0.04384 | −0.00060 | [−0.00142, 0.00022] | 0.138 |
| 冲突 | 0.04070 | 0.03928 | −0.00142 | [−0.00300, 0.00015] | 0.074 |
| 强冲突 | 0.05036 | 0.04977 | −0.00059 | [−0.00331, 0.00215] | 0.682 |
| 可判定非冲突 | 0.03204 | 0.03056 | −0.00149 | [−0.00354, 0.00052] | 0.148 |

- Base candidate recall@200：seed 2027/2028/2029 为 `0.66205/0.65842/0.67053`，均值 `0.66367`；Jev 不改变该数值。
- validation 选择标准仍是预先设定的“冲突 NDCG@10，其次总体 NDCG@10”。seed2028/2029 的最优阈值 `0.4` 只在约 4% case 上启用 Jev，其余回退 Base；因此必须同时报告 coverage，不能把接近 Base 的结果解释为全量 Jev 有效。
- 全量不设阈值的敏感性结果更差：alpha=`0.75` 的三 seed 平均总体 NDCG@10 为 `0.04159`，alpha=`0.5` 为 `0.03779`，Jev-only（alpha=`0`）为 `0.02945`。

### 正式结果结论

当前证据支持：完整 SASRec 的候选池质量明显高于开发版，hosted Jev 在该 MovieLens 管线上没有带来总体、冲突或强冲突的显著收益；它也没有改善召回。开发版的正增益不能外推为完整 SASRec 的主结果。Jev 的潜在价值仍需在 IDNP 或带可靠文本 metadata 的数据集上，通过更稳定的置信度校准、覆盖率约束和蒸馏方案重新验证。

### Hosted Jev 调用审计

- 真实调用总数：34,674；实际返回模型：`jev-1.13.0`；成功率 100%。
- 输入 token：264,718,450；输出 token：70,076,154；成功请求延迟中位数 989.7 ms，P95 2,216.6 ms。
- confidence 中位数 0.15；`confidence >= 0.4` 的比例仅 4.27%，解释了阈值策略的低 coverage。

### 历史开发版 Jev 结果（不作为正式主结果）

- 数据：MovieLens-1M 正反馈，5,779 名用户、每 seed 5,779 个 test case。三个 seed 使用相同用户集合；统计推断先在用户内平均 seed 结果，再按用户 bootstrap 和 paired sign permutation。
- Jev：validation 对三个 seed 均选择 `alpha=0`、无置信度阈值；test 固定此设置。调用模型实际版本为 `jev-1.13.0`。
- 当前 SASRec 对照是 `SASRec-dev`（`maxlen=50`、10 epochs），不是完整正式训练配置；下表属于开发阶段证据，不作为最终论文主结果。

| Seed | SASRec-dev NDCG@10 | SASRec-dev + Jev | Δ | 冲突 NDCG@10：base → Jev | 强冲突 NDCG@10：base → Jev | Candidate recall@200 |
|---:|---:|---:|---:|---:|---:|---:|
| 2027 | 0.02035 | 0.02793 | +0.00758 | 0.01832 → 0.02417 | 0.02495 → 0.03104 | 0.42101 |
| 2028 | 0.01947 | 0.02827 | +0.00881 | 0.01635 → 0.02802 | 0.01506 → 0.04238 | 0.43381 |
| 2029 | 0.01843 | 0.02892 | +0.01048 | 0.01221 → 0.02320 | 0.01584 → 0.02804 | 0.42533 |

### 用户级多 seed 配对统计

| 分组 | 用户数 | SASRec-dev NDCG@10 | 加 Jev | Δ | 95% 用户 bootstrap CI | paired permutation p |
|---|---:|---:|---:|---:|---:|---:|
| 全部 | 5,779 | 0.01942 | 0.02837 | +0.00896 | [0.00574, 0.01207] | 0.00020 |
| 冲突 | 1,175 | 0.01563 | 0.02513 | +0.00951 | [0.00391, 0.01564] | 0.00260 |
| 强冲突 | 509 | 0.01862 | 0.03382 | +0.01520 | [0.00440, 0.02654] | 0.00560 |
| 可判定非冲突 | 593 | 0.01772 | 0.01730 | −0.00042 | [−0.00866, 0.00804] | 0.92821 |

- 三个 seed 的整体增益方向一致；按 seed 分别检验时，冲突组在 2028、2029 显著，2027 未显著。合并后按用户聚类的冲突与强冲突提升显著，非冲突组无显著变化。
- Jev 只重排 SASRec 给出的 Top-200 候选，不改变召回。平均 candidate recall@200 为 0.42672；候选内 conditional NDCG@10 均值从 0.04551 升至 0.06649。
- 三 seed 每个用户的冲突标签一致；冲突组和可判定非冲突组没有用户交集，因此同一用户内的“冲突增益减非冲突增益”不可估计。

### 与状态启发式的诊断对照

在相同 Top-200 候选池上，使用 item genre 构造 `recent-only`、`long-only`、`equal` 和基于 JSD 的 `adaptive` 探针。它们是诊断对照，不是训练模型。

| 对照（3-seed 均值） | NDCG@10 | 冲突 NDCG@10 | 强冲突 NDCG@10 |
|---|---:|---:|---:|
| recent-only | 0.02641 | 0.02723 | 0.02409 |
| long-only | 0.01811 | 0.00758 | 0.00436 |
| equal | 0.02503 | 0.02012 | 0.01685 |
| adaptive | 0.02586 | 0.02547 | 0.02458 |
| SASRec-dev + Jev | 0.02837 | 0.02513 | 0.03382 |

- Jev 相对 `recent-only` 的用户级差值为：总体 +0.00196（95% CI [−0.00148, 0.00542], p=0.256），冲突 −0.00210（[−0.00907, 0.00527], p=0.591），强冲突 +0.00973（[−0.00197, 0.02171], p=0.101）。当前证据不能声称 Jev 普遍优于近期偏好启发式。
- Jev 相对 `equal` 在强冲突上的差值为 +0.01697（95% CI [0.00602, 0.02873], p=0.004）；相对 `adaptive` 的强冲突差值为 +0.00924（[−0.00191, 0.02115], p=0.103）。
- 当前最稳妥结论是：Jev 能显著改善 SASRec-dev 的候选内排序，收益在强冲突中最明显；相对于强的近期状态启发式，额外收益仍未被充分证明。

### 效率与失败模式

- validation + test 合计 34,674 次 Jev 调用，成功率 100%；输入约 265.9M tokens，输出约 70.1M tokens；API 成功延迟中位数 1.109 秒，P95 为 2.083 秒。
- 长尾目标（训练频次 ≤5）每 seed 41 个，全部不在 Top-200 候选中，Jev 无法挽回；当前瓶颈仍包括候选召回。
- 案例仅作机制示例：seed2028 强冲突 case `test_231` 的 Toy Story 从 SASRec 第 34 位升至 Jev 第 1 位（confidence 0.29）；`test_1904` 的 Titanic 从第 1 位降至第 37 位（confidence 0.10）。
- MovieLens session 由评分时间间隔构造，是行为序列代理，不等同于真实浏览或购买 session。

## 当前阶段与下一步

1. 三 seed full-SASRec 训练、Jev validation/test cache、冻结调参和配对统计均已完成；正式结果见上文。
2. 当前不能把 hosted Jev 的收益写成已验证结论：full-SASRec 上主比较为不显著且方向接近零，低 confidence coverage 是主要限制。
3. 下一步接入 IDNP 官方实现；随后在带可靠文本 metadata 的 Amazon/Movies & TV 数据上重复相同协议，并加入覆盖率约束、confidence calibration、候选顺序鲁棒性和本地蒸馏对照。

## 历史 P0/P1 执行状态（2026-10-11）

## P0 数据管线：完成

- 数据：MovieLens-1M，评分 `>=4`。
- 原始评分：1,000,209；正反馈：575,281。
- 用户：5,779；训练物品：3,110；训练交互：553,326。
- valid/test 各 5,779 条。
- valid/test 冲突样本：1,180 / 1,175。
- valid/test 强冲突样本：489 / 509。
- `future_in_prefix=0`，`overlapping_train_holdout_timestamps=0`。
- 解释边界：MovieLens 的 session 是评分时间间隔代理，不能直接解释为浏览或购买 session。

## P1 开发运行：部分完成

### 无效运行

`results/dev/sasrec_seed2027_smoke`、`sasrec_seed2027_len50`、`sasrec_seed2028_len50` 使用了错误的左填充状态索引，已由 `INVALID_padding_bug.txt` 标记，所有数字均不进入证据表。

### 修正版开发运行

实现：`scripts/train_sasrec.py`，修复左填充后取最后真实行为的位置；配置：`configs/ml1m_dev.yaml`，`maxlen=50`、10 epochs，开发配置，不是最终论文配置。

| Seed | Split | NDCG@10 | Conflict NDCG@10 | Strong-conflict NDCG@10 | Candidate recall@200 | Conflict recall@200 |
|---:|---|---:|---:|---:|---:|---:|
| 2027 | test | 0.02035 | 0.01832 | 0.02495 | 0.42101 | 0.35915 |
| 2028 | test | 0.01947 | 0.01635 | 0.01506 | 0.43381 | 0.38128 |
| 2029 | test | 0.01843 | 0.01221 | 0.01584 | 0.42533 | 0.36766 |

这些数字只用于确认管线和训练稳定性，不能写成论文主结果。候选召回仍低于计划中的 0.5 门槛；Jev 实验必须报告全量和 candidate-retrieved 条件结果。

## P3 前置：完成

- 已生成 Jev 结构化请求脚本和 3 条 validation dry-run 请求。
- 每条请求含 long/recent 分布、时间间隔、行为类型、冲突标签和 200 个候选。
- `api_called=false`；没有发送外部请求。

## 状态探针：完成（诊断，不是主基线）

在 seed 2027 修正版 SASRec 的同一 Top-200 候选池上，用 item genre 与状态分布做不训练的匹配探针：

| Probe | Test NDCG@10 | Conflict NDCG@10 | Strong-conflict NDCG@10 |
|---|---:|---:|---:|
| Recent-only | 0.02655 | 0.02470 | 0.02102 |
| Long-only | 0.01771 | 0.00722 | 0.00380 |
| Equal long/recent | 0.02512 | 0.01942 | 0.01589 |
| JSD adaptive heuristic | 0.02577 | 0.02380 | 0.02242 |

这些值说明冲突子集中的近期信号具有诊断价值；它们不代表已训练模型或 Jev 结果。

## 历史阻塞记录（首次真实 Jev 调用前）

- Jev 调用 endpoint 已切换为用户提供的 `https://ai.xiaohaima-edu.com/v1/systemone`，模型为 `jev-latest`；A5000 的 `llm4rec` 中仍未读到 `JEV_API_KEY`。尚未发出真实 API 请求，额度与认证可用性未验证。
- 请求缓存复查发现：`state.recent_movie_ids` 是 MovieLens 原始 movie ID，生成请求时错误地按编码后的 item ID 查标题。须按 `items.json` 的 `movie_id` 字段修正近期历史 metadata，然后重新生成请求缓存及 hash。该问题不涉及已报告的 SASRec 排名与类别匹配探针，但现有 Jev 请求缓存不能用于真实调用。
- Amazon Reviews'23 尚未下载；MovieLens 结果应视为管线验证，不应承担最终语义冲突结论。

## 原执行计划（已由上方最新进度更新）

1. 修正 Jev 请求的近期电影 ID/标题映射，检查近期电影类别与 recent distribution 一致，重新生成请求缓存及 hash。
2. 核对正式基线的实现与完整配置，补足训练并改善 Top-200 召回；当前 3 个 seed 的 10 epoch 开发运行已全部完成，最佳 epoch 均为 10。
3. 密钥可用后先做 3 条真实接口检查；通过后在 validation 调融合权重和置信度阈值，冻结参数后评估 test。
4. 接入 IDNP 官方实现，运行 IDNP、相同候选池的 IDNP + Jev 及 long/recent/gap/fusion 消融。
5. 接入 Amazon Books 或 Movies & TV 的文本 metadata 主验证集，再完成统计显著性、多样性、长尾与鲁棒性实验。

## 2026-10-11 P3 请求缓存与接口准备

- 已将请求体对齐 TypeSafe `POST /v1/systemone`：`state`、固定版本 `jev-1.13.0`、`questions.next_item` 和最多 200 个候选 `criteria`。
- `state` 中的长期偏好和近期会话分布已带 genre 名称；时间间隔、当前 session 长度、评分行为和冲突标签均保留。
- 候选的 SASRec 分数只保留在 `audit.base_scores`，不发送给 Jev；离线融合仍使用冻结的 base score。
- A5000 已生成 validation request cache：`results/jev_requests/valid_seed2027.jsonl`、`valid_seed2028.jsonl`、`valid_seed2029.jsonl`，每个 5,779 条；缓存生成阶段 `api_called=false`。
- 当前缓存 SHA-256：
  - seed2027 `10993d0d007a03c3c0aa5a08a862a3dd49143b01ab64e5c5907afeca6643d811`
  - seed2028 `dbd065cf0ceba76c7cec3a2c0b6ea17877cc6fb91abb858556f7b4f70371b16c`
  - seed2029 `b176838c6853aac119ce3754414e284bdf5e8a1015ecb3755c0d084f304068fa`
- A5000 检查仍显示 `JEV_API_KEY_ABSENT`；真实请求脚本在缺少 key 时退出且不创建输出文件。

## 2026-10-11 配置与仓库约定

- 后续处理数据固定放在 `/media/sdust307/disk3/lyk/jev_seqrec_experiment/data`；数据、模型权重、候选 JSONL 和日志均由 Git 忽略。
- Jev 调用配置改为环境变量：`JEV_ENDPOINT=https://ai.xiaohaima-edu.com/v1/systemone`、`JEV_MODEL=jev-latest`、`JEV_API_KEY`。key 不写入仓库或实验目录。
- 请求构建脚本已修正 `recent_movie_ids` 的 MovieLens 原始 `movie_id` 与编码 `item_id` 的命名空间映射；旧请求缓存需重建后才能调用。
- 本地代码仓库根目录为 `D:\PyProject\Jev\jev_seqrec_experiment`，A5000 工作树为 `/media/sdust307/disk3/lyk/jev_seqrec_experiment`；两者通过 Git 保持代码一致，数据仍留在 A5000 的 `data/`。

## 2026-10-11 Jev 凭据与接口 smoke

- A5000 `llm4rec` 已检测到 `JEV_API_KEY_PRESENT`；key 内容未输出、未写入仓库。
- endpoint：`https://ai.xiaohaima-edu.com/v1/systemone`；请求模型：`jev-latest`；服务返回的实际模型：`jev-1.13.0`。
- 3 条 validation smoke 请求全部成功，均包含 200 个候选，错误数为 0；服务返回了完整概率分布、confidence 和 token usage。
- smoke 的端到端延迟约为 9.7–29.9 秒；这只是接口连通性检查，不能作为最终延迟结论。
- 下一步可生成完整 validation cache；随后在 validation 冻结融合权重和 confidence threshold，再评估 test。

## 当前文件位置与证据范围

- A5000 项目根目录：`/media/sdust307/disk3/lyk/jev_seqrec_experiment`。
- 原始 MovieLens 压缩包：`/media/sdust307/disk2/lyk/jev_data_raw/ml-1m.zip`。
- 处理后数据与审计：项目根目录下 `data/ml1m/`。
- 训练实现、数据构建、状态探针、Jev 请求/调用/离线评价：`scripts/`；共享状态和指标协议：`src/protocol.py`；配置：`configs/`。
- 有效 SASRec 开发运行：`results/dev/sasrec_seed2027_len50_fixed/`、`sasrec_seed2028_len50_fixed/`、`sasrec_seed2029_len50_fixed/`。每个目录含 `model.pt`、`train_log.json`、`run_manifest.json`、valid/test 候选与指标 JSON。
- checkpoint 目前实际保存在上述结果目录的 `model.pt`；配置里预留的 `/media/sdust307/disk2/lyk/jev_checkpoints` 当前为空。
- 状态探针：`results/dev/state_probe_seed2027/`；popularity/random：`results/dev/heuristics/`；训练日志：`logs/`。
- Jev validation 请求：`results/jev_requests/`，3 个 seed 各 5,779 条，均未调用 API，且待修正 metadata 映射后重建。
- 本地代码与方案副本：`D:\PyProject\Jev`。完整数据、checkpoint 与候选结果目前在服务器，本地没有完整结果镜像。
- IDNP、BERT4Rec、学习型 long/recent gate、Jev 正式评价、统计显著性、多样性和完整冷启动/长尾分析尚未完成。
