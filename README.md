# JEV × 序列推荐实验

本目录是序列推荐实验的代码仓库。数据和生成结果不进入 Git，服务器上的数据统一放在：

```text
/media/sdust307/disk3/lyk/jev_seqrec_experiment/data
```

## Jev 配置

只通过运行环境注入凭据和接口配置，不把 key 写入文件：

```bash
conda env config vars set -n llm4rec \
  JEV_API_KEY='在服务器终端交互输入的 key' \
  JEV_ENDPOINT='https://ai.xiaohaima-edu.com/v1/systemone' \
  JEV_MODEL='jev-latest'
```

设置后重新打开 shell 或执行 `conda activate llm4rec`。检查时只输出 `PRESENT/ABSENT`，不要打印变量值：

```bash
conda run -n llm4rec python -c "import os; print('PRESENT' if os.getenv('JEV_API_KEY') else 'ABSENT')"
```

`scripts/run_jev_seqrec.py` 从 `JEV_API_KEY`、`JEV_ENDPOINT`、`JEV_MODEL` 读取配置；缺少 key 时不会发送请求。

## 运行入口

```bash
conda activate llm4rec
python scripts/prepare_ml1m.py --config configs/ml1m.yaml
python scripts/prepare_jev_requests.py \
  --candidates results/dev/sasrec_seed2027_len50_fixed/valid_candidates.jsonl \
  --out results/jev_requests/valid_seed2027.jsonl
python scripts/run_jev_seqrec.py \
  --requests results/jev_requests/valid_seed2027.jsonl \
  --out results/jev_responses/valid_seed2027.jsonl --limit 3
```

完整实验状态见 `results/dev_STATUS.md`。正式结果必须区分候选召回和候选集内重排效果。
