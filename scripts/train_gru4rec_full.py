"""Train and evaluate a reproducible GRU4Rec-style recurrent baseline.

This is an explicitly configured BCE recurrent baseline, not a claim of exact
reproduction of every GRU4Rec implementation.  It shares the frozen split,
full-catalog evaluator, negative-sampling policy, and early-stopping protocol
with the full SASRec run.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from protocol import read_config, sha256, write_json  # noqa: E402
from catalog_evaluation import evaluate_catalog, read_jsonl, result_metrics, write_records  # noqa: E402


class GRU4RecStyle(nn.Module):
    def __init__(self, n_items, hidden_units, num_layers=1, dropout_rate=0.0):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, hidden_units, padding_idx=0)
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout_rate if num_layers > 1 else 0.0)
        self.out_emb = nn.Embedding(n_items + 1, hidden_units, padding_idx=0)
        self.out_bias = nn.Parameter(torch.zeros(n_items + 1))
        self.n_items = n_items

    def encode_right_padded(self, seqs, lengths):
        lengths = lengths.to(dtype=torch.long)
        embedded = self.item_emb(seqs)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        output, _ = pad_packed_sequence(packed_out, batch_first=True,
                                        total_length=seqs.shape[1])
        return output

    def forward(self, seqs, pos, neg, lengths):
        hidden = self.encode_right_padded(seqs, lengths)
        pos_scores = (hidden * self.out_emb(pos)).sum(-1) + self.out_bias[pos]
        neg_scores = (hidden * self.out_emb(neg)).sum(-1) + self.out_bias[neg]
        return pos_scores, neg_scores

    def score_catalog(self, left_padded_seqs):
        # catalog_evaluation supplies left-padded histories. Convert them to
        # right-padded sequences so packed GRU output has an unambiguous last
        # valid state for every row.
        lengths = (left_padded_seqs != 0).sum(axis=1).astype(np.int64)
        maxlen = left_padded_seqs.shape[1]
        right = np.zeros_like(left_padded_seqs)
        for i, length in enumerate(lengths):
            if length:
                right[i, :length] = left_padded_seqs[i, -length:]
        seqs = torch.as_tensor(right, dtype=torch.long, device=self.item_emb.weight.device)
        lengths_t = torch.as_tensor(np.maximum(lengths, 1), dtype=torch.long,
                                    device=seqs.device)
        hidden = self.encode_right_padded(seqs, lengths_t)
        last = hidden[torch.arange(seqs.shape[0], device=seqs.device), lengths_t - 1]
        return last @ self.out_emb.weight[1:].T + self.out_bias[1:]


def sample_sequences(users, indices, maxlen, n_items, rng):
    seq = np.zeros((len(indices), maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    lengths = np.zeros(len(indices), dtype=np.int64)
    for b, i in enumerate(indices):
        items = users[i]["items"]
        length = min(maxlen, len(items) - 1)
        lengths[b] = length
        seq[b, :length] = items[-length - 1:-1]
        pos[b, :length] = items[-length:]
        excluded = set(items)
        values = rng.integers(1, n_items + 1, size=length)
        bad = np.asarray([v in excluded for v in values])
        while bad.any():
            values[bad] = rng.integers(1, n_items + 1, size=int(bad.sum()))
            bad = np.asarray([v in excluded for v in values])
        neg[b, :length] = values
    return seq, pos, neg, lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ml1m_gru4rec.yaml")
    ap.add_argument("--data", default="data/ml1m")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = read_config(ROOT / args.config)
    mc = cfg["model"]
    torch.set_num_threads(mc["torch_threads"])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_per_process_memory_fraction(
            mc["gpu_memory_limit_gib"] * 1024**3 /
            torch.cuda.get_device_properties(0).total_memory)
    if mc["device"] == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required by this configuration")
    device = mc["device"]
    data = ROOT / args.data
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run_manifest.json").exists() and not args.resume:
        raise SystemExit("Existing run: use --resume rather than overwrite")
    users = read_jsonl(data / "train.jsonl")
    valid_rows = read_jsonl(data / "valid.jsonl")
    n_items = len(json.loads((data / "items.json").read_text(encoding="utf-8")))
    model_args = SimpleNamespace(**mc)
    model = GRU4RecStyle(n_items, model_args.hidden_units, model_args.num_layers,
                         model_args.dropout_rate).to(device)
    nn.init.xavier_normal_(model.item_emb.weight.data[1:])
    nn.init.xavier_normal_(model.out_emb.weight.data[1:])
    model.item_emb.weight.data[0].zero_()
    model.out_emb.weight.data[0].zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=mc["learning_rate"],
                                 betas=tuple(mc["adam_betas"]))
    rng = np.random.default_rng(args.seed)
    log, best, best_epoch, stale, start_epoch = [], -math.inf, 0, 0, 1
    if args.resume and (out / "last.pt").exists():
        last = torch.load(out / "last.pt", map_location=device, weights_only=False)
        if last["seed"] != args.seed or last["config"] != cfg:
            raise ValueError("Resume seed or configuration changed")
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        rng.bit_generator.state = last["numpy_rng"]
        torch.set_rng_state(last["torch_rng"].cpu())
        if last.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(last["cuda_rng"])
        log, best, best_epoch, stale = (last["log"], last["best"],
                                        last["best_epoch"], last["stale"])
        start_epoch = last["epoch"] + 1
    manifest = {
        "method": "GRU4Rec-style-BCE", "seed": args.seed, "config": cfg,
        "model_source_sha256": sha256(Path(__file__)),
        "dataset_hashes": json.loads((data / "hashes.json").read_text(encoding="utf-8")),
        "selection": "full validation NDCG@10", "test_used_for_selection": False,
        "status": "running", "started_at": time.time(),
    }
    write_json(out / "run_manifest.json", manifest)
    stopping_reason = "max_epochs"
    for epoch in range(start_epoch, mc["max_epochs"] + 1):
        started = time.perf_counter()
        model.train()
        losses = []
        order = rng.permutation(len(users))
        for s in range(0, len(users), mc["batch_size"]):
            idx = order[s:s + mc["batch_size"]]
            seq, pos, neg, lengths = sample_sequences(users, idx, mc["maxlen"],
                                                       n_items, rng)
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            lengths_t = torch.as_tensor(lengths, dtype=torch.long, device=device)
            positive, negative = model(seq_t, pos_t, neg_t, lengths_t)
            keep = pos_t > 0
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                        positive[keep], torch.ones_like(positive[keep])) +
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        negative[keep], torch.zeros_like(negative[keep])))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        entry = {"epoch": epoch, "loss": float(np.mean(losses)),
                 "seconds": time.perf_counter() - started}
        if epoch % mc["eval_every"] == 0 or epoch == mc["max_epochs"]:
            metrics = result_metrics(evaluate_catalog(
                model, valid_rows, mc["maxlen"], batch_size=mc["eval_batch_size"],
                exclude_seen=mc["exclude_seen"]))
            entry["valid"] = metrics
            score = metrics["all"]["ndcg@10"]
            if score > best:
                best, best_epoch, stale = score, epoch, 0
                torch.save({"model": model.state_dict(), "config": cfg,
                            "seed": args.seed, "best_epoch": epoch,
                            "best_valid_ndcg10": best}, out / "model.pt")
            else:
                stale += 1
            log.append(entry)
            write_json(out / "train_log.json", {
                "best_epoch": best_epoch, "best_valid_ndcg10": best,
                "history": log})
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "config": cfg, "seed": args.seed, "epoch": epoch, "log": log,
                        "best": best, "best_epoch": best_epoch, "stale": stale,
                        "numpy_rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None},
                       out / "last.pt")
            print(json.dumps({"epoch": epoch, "loss": entry["loss"],
                              "valid_ndcg10": score,
                              "candidate_recall200": metrics["all"]["candidate_recall@200"],
                              "stale": stale}), flush=True)
            if epoch >= mc["min_epochs"] and stale >= mc["patience_evaluations"]:
                stopping_reason = "validation_patience"
                break
        else:
            log.append(entry)
    best_state = torch.load(out / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    for split in ("valid", "test"):
        rows = valid_rows if split == "valid" else read_jsonl(data / "test.jsonl")
        records = evaluate_catalog(model, rows, mc["maxlen"],
                                   batch_size=mc["eval_batch_size"],
                                   exclude_seen=mc["exclude_seen"])
        write_records(out / f"{split}_candidates.jsonl", records)
        write_json(out / f"{split}_metrics.json", result_metrics(records))
    manifest.update({"status": "complete", "best_epoch": best_epoch,
                     "epochs_ran": log[-1]["epoch"],
                     "stopping_reason": stopping_reason, "finished_at": time.time(),
                     "checkpoint_sha256": sha256(out / "model.pt")})
    write_json(out / "run_manifest.json", manifest)
    print(json.dumps({"status": "complete", "best_epoch": best_epoch,
                      "out": args.out}), flush=True)


if __name__ == "__main__":
    main()
