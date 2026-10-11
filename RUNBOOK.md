# 执行 Runbook：P0 → P3

## 当前状态

- 实验目录：`/media/sdust307/disk3/lyk/jev_seqrec_experiment`
- 环境：`llm4rec`（Python 3.9.23、PyTorch 2.5.1、CUDA 11.8）
- 最近复查：A5000 总显存 24564 MiB，已用约 12497 MiB，可用约 11736 MiB；仍有共享任务运行，不要杀进程。
- Jev：使用环境变量 `JEV_API_KEY`、`JEV_ENDPOINT`、`JEV_MODEL`；key 不写入仓库或实验目录。

## 固定路径与 Jev 配置

- 处理后数据统一放在：`/media/sdust307/disk3/lyk/jev_seqrec_experiment/data`。
- 当前接口：`https://ai.xiaohaima-edu.com/v1/systemone`。
- 当前模型：`jev-latest`。
- 在 A5000 交互式终端安全注入共享 key：

```bash
read -rsp "Jev API key: " JEV_KEY; echo
conda env config vars set -n llm4rec \
  JEV_API_KEY="$JEV_KEY" \
  JEV_ENDPOINT="https://ai.xiaohaima-edu.com/v1/systemone" \
  JEV_MODEL="jev-latest" >/dev/null
unset JEV_KEY
conda activate llm4rec
```

只检查存在性，不打印 key：

```bash
conda run -n llm4rec python -c "import os; print('JEV_API_KEY_PRESENT' if os.getenv('JEV_API_KEY') else 'JEV_API_KEY_ABSENT')"
```

## P0：数据与冲突管线

首轮只做 MovieLens-1M，避免 Amazon 原始数据占满 disk3。

```bash
conda activate llm4rec
cd /media/sdust307/disk3/lyk/jev_seqrec_experiment
mkdir -p /media/sdust307/disk2/lyk/jev_data_raw /media/sdust307/disk3/lyk/jev_seqrec_experiment/data/ml1m
```

下载地址使用 GroupLens 官方页：<https://grouplens.org/datasets/movielens/1m/>。下载后：

1. 解析 `ratings.dat`，按 user/timestamp 排序；
2. 固定 leave-last-two split；
3. 生成 session（gap=30min）；
4. 生成 long/recent 分布、JSD 冲突标签和 `data/conflict_manifest.csv`；
5. 保存 `dataset_stats.json`、`split_hash.txt` 和 `conflict_manifest_hash.txt`。

P0 验收条件：无未来信息泄漏；每个测试样本有 prefix；冲突 manifest 可复现；输出用户数、物品数、交互数、session 数、冲突比例和缺失元数据比例。

## P1：本地基线

先用 seed=2027、candidate N=20 做数据流 smoke，然后运行 3 seeds：2027/2028/2029。

顺序：

1. popularity/random sanity；
2. SASRec；
3. BERT4Rec；
4. IDNP 官方实现（若完整依赖或 checkpoint 不可用，记录 blocked，不用简化版冒充主结果）。

P1 验收条件：candidate recall@20、Recall@10、NDCG@10、冲突 NDCG@10 均有 JSON 输出；训练/验证/测试均使用同一切分。

## P2：机制变体

在 MovieLens-1M 和 Amazon Books 小规模子集运行：

```text
short-only
long-only
long+recent concat
long/recent gated
+time gap
+behavior type
full local state
```

冻结冲突阈值和主要指标后再扩大候选规模，避免在测试集调参。

## P3：Jev cache 与重排

需要先把凭据放入受控环境变量，不写入目录：

```bash
export JEV_API_KEY='通过服务器 secret 注入；不要写入命令历史或仓库'
```

调用流程：

1. validation 预计算 Jev response cache；
2. 固定模型版本、prompt hash、候选顺序和 schema；
3. 在 validation 选择融合权重 alpha 与 confidence threshold；
4. 冻结 cache 后评估 test；
5. 记录 latency、timeout、retry、cost、fallback 和 cache hit。

主比较：`IDNP` vs `IDNP + Jev`，主指标为冲突子集 NDCG@10；同时报告总体指标、长尾、多样性、ECE 和 p95 latency。

## 停止条件

- candidate recall@200 < 0.5：先修召回，暂停解释 Jev 重排；
- Jev error/timeout > 5%：先修 cache/fallback；
- 冲突样本少于 200 个用户：只做描述性分析，不能做显著性结论；
- GPU 可用显存不足：降低 batch/序列长度或等待共享任务结束，不改变方法定义。

## 交付物顺序

`dataset_stats.json` → `conflict_manifest.csv` → `baseline_results.json` → `ablation_results.json` → `jev_cache/` → `main_results.json` → `stats_report.json` → 论文表格。
