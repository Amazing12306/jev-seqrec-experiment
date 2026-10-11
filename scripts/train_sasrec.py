"""Train a reproducible local SASRec-style next-item ranker on ML-1M.

This is a development implementation for the P1 gate. It is deliberately
separate from the final comparison until the full schedule and configuration
are frozen. The training target is always a prefix next item, so no held-out
timestamp is visible to the model.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from protocol import read_config, rank_metrics, sliced_metrics, write_json


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pad_batch(histories, maxlen, device):
    values = np.zeros((len(histories), maxlen), dtype=np.int64)
    for row, history in enumerate(histories):
        seq = history[-maxlen:]
        if seq:
            values[row, -len(seq):] = seq
    return torch.as_tensor(values, dtype=torch.long, device=device)


class CausalSASRec(nn.Module):
    def __init__(self, n_items, maxlen, hidden, blocks, heads, dropout):
        super().__init__()
        self.n_items = n_items
        self.maxlen = maxlen
        self.item = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.position = nn.Embedding(maxlen, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=blocks)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def encode(self, histories):
        bsz, length = histories.shape
        pos = torch.arange(length, device=histories.device)[None, :]
        x = self.dropout(self.item(histories) + self.position(pos))
        padding = histories.eq(0)
        # The input consists only of a prefix. Causal masking prevents the
        # intermediate training states from seeing later prefix events.
        causal = torch.triu(torch.ones(length, length, device=histories.device, dtype=torch.bool), diagonal=1)
        x = self.encoder(x, mask=causal, src_key_padding_mask=padding)
        x = self.norm(x)
        # Histories are left-padded. The last real event is therefore at the
        # final column for every non-empty prefix; indexing by the count would
        # incorrectly select an early/padding position for short histories.
        return x[:, -1, :]

    def score(self, histories, item_ids):
        user = self.encode(histories)
        if item_ids.ndim == 1:
            item_ids = item_ids[:, None]
        return torch.einsum("bd,bkd->bk", user, self.item(item_ids))

    def score_all(self, histories):
        user = self.encode(histories)
        return user @ self.item.weight[1:].T


def training_rows(users):
    rows = []
    for user in users:
        seq = list(map(int, user["items"]))
        for idx in range(1, len(seq)):
            rows.append((seq[:idx], seq[idx]))
    return rows


def evaluate(model, rows, maxlen, device, n_items, candidate_limit=200, batch_size=256):
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            histories = pad_batch([r["history"] for r in batch], maxlen, device)
            scores = model.score_all(histories)
            top_n = min(candidate_limit, n_items)
            values, indices = torch.topk(scores, k=top_n, dim=1)
            values = values.detach().cpu().numpy()
            indices = indices.detach().cpu().numpy() + 1
            for local_idx, (row, ids, vals) in enumerate(zip(batch, indices, values)):
                target = int(row["target_id"])
                rank_all = int((scores[local_idx] > scores[local_idx, target - 1]).sum().item() + 1) if target else 0
                rank_candidate = int(np.where(ids == target)[0][0] + 1) if target and target in ids else 0
                rank = rank_all if target else 0
                record = {
                    "case_id": row["case_id"], "user_id": row["user_id"], "split": row["split"],
                    "target_id": target, "target_known": bool(row["target_known"]),
                    "rank": rank, "rank_candidate": rank_candidate,
                    "candidate_ids": ids.tolist(), "candidate_scores": vals.tolist(),
                    "conflict": bool(row["state"]["conflict"]),
                    "conflict_strength": row["state"]["conflict_strength"],
                    "target_timestamp_group_size": int(row["target_timestamp_group_size"]),
                    "target_train_frequency": int(row["target_train_frequency"]),
                    "train_length": int(row["train_length"]),
                    "state": row["state"],
                }
                output.append(record)
    return output


def metrics(records):
    result = sliced_metrics(records)
    result["candidate_recall@200_full"] = float(np.mean([r["rank_candidate"] > 0 for r in records])) if records else 0.0
    result["unknown_target_n"] = sum(not r["target_known"] for r in records)
    result["tie_target_n"] = sum(r["target_timestamp_group_size"] > 1 for r in records)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ml1m.yaml")
    ap.add_argument("--data", default="data/ml1m")
    ap.add_argument("--out", default="results/dev/sasrec_seed2027")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = read_config(root / args.config)
    data = root / args.data
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_items = len(json.loads((data / "items.json").read_text(encoding="utf-8")))
    train_users = read_jsonl(data / "train.jsonl")
    valid_rows = read_jsonl(data / "valid.jsonl")
    test_rows = read_jsonl(data / "test.jsonl")
    train = training_rows(train_users)
    model_cfg = cfg["model"]
    model = CausalSASRec(n_items, model_cfg["maxlen"], model_cfg["hidden_units"],
                         model_cfg["num_blocks"], model_cfg["num_heads"], model_cfg["dropout_rate"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_cfg["learning_rate"],
                                  betas=tuple(model_cfg["adam_betas"]), weight_decay=model_cfg["l2_emb"])
    epochs = args.epochs or model_cfg["max_epochs"]
    batch_size = args.batch_size or model_cfg["batch_size"]
    history = []
    best = -math.inf
    best_epoch = 0
    stale = 0
    rng = random.Random(args.seed)
    for epoch in range(1, epochs + 1):
        rng.shuffle(train)
        model.train()
        losses = []
        for start in range(0, len(train), batch_size):
            batch = train[start:start + batch_size]
            histories = pad_batch([x[0] for x in batch], model_cfg["maxlen"], device)
            pos = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
            neg = torch.randint(1, n_items + 1, pos.shape, device=device)
            neg = torch.where(neg.eq(pos), (neg % n_items).clamp_min(1), neg)
            logits = model.score(histories, torch.stack([pos, neg], dim=1))
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, torch.tensor([[1.0, 0.0]], device=device).expand_as(logits))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        # Validation only on a deterministic prefix sample to keep the smoke bounded.
        eval_sample = valid_rows
        valid_records = evaluate(model, eval_sample, model_cfg["maxlen"], device, n_items, 200)
        valid_metrics = metrics(valid_records)
        score = valid_metrics["all"].get("ndcg@10", 0.0)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "valid": valid_metrics})
        print(json.dumps({"epoch": epoch, "loss": float(np.mean(losses)), "valid_ndcg@10": score}), flush=True)
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            torch.save({"model": model.state_dict(), "config": cfg, "seed": args.seed,
                        "n_items": n_items, "best_epoch": epoch}, out / "model.pt")
        else:
            stale += 1
        if epoch >= model_cfg["min_epochs"] and stale >= model_cfg["patience_evaluations"]:
            break
    write_json(out / "train_log.json", {"seed": args.seed, "device": str(device), "epochs_ran": len(history),
                                        "best_epoch": best_epoch, "best_valid_ndcg@10": best, "history": history})
    checkpoint = torch.load(out / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    for split, rows in (("valid", valid_rows), ("test", test_rows)):
        records = evaluate(model, rows, model_cfg["maxlen"], device, n_items, 200)
        with (out / f"{split}_candidates.jsonl").open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_json(out / f"{split}_metrics.json", metrics(records))
    write_json(out / "run_manifest.json", {"method": "SASRec-dev", "seed": args.seed,
        "data_dir": str(data), "epochs": len(history), "device": str(device),
        "status": "development_run; not publication result"})
    print(json.dumps({"out": str(out), "best_epoch": best_epoch, "best_valid_ndcg@10": best}))


if __name__ == "__main__":
    main()
