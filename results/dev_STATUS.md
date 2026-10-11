# P0/P1 执行状态（2026-10-11）

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

## 当前阻塞

- Jev 调用 endpoint 已切换为用户提供的 `https://ai.xiaohaima-edu.com/v1/systemone`，模型为 `jev-latest`；A5000 的 `llm4rec` 中仍未读到 `JEV_API_KEY`。尚未发出真实 API 请求，额度与认证可用性未验证。
- 请求缓存复查发现：`state.recent_movie_ids` 是 MovieLens 原始 movie ID，生成请求时错误地按编码后的 item ID 查标题。须按 `items.json` 的 `movie_id` 字段修正近期历史 metadata，然后重新生成请求缓存及 hash。该问题不涉及已报告的 SASRec 排名与类别匹配探针，但现有 Jev 请求缓存不能用于真实调用。
- Amazon Reviews'23 尚未下载；MovieLens 结果应视为管线验证，不应承担最终语义冲突结论。

## 下一步

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
