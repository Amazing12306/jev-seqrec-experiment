"""Train the pinned, complete upstream SASRec model on the frozen splits."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "sasrec_pytorch"))
from model import SASRec
from protocol import read_config, sha256, write_json
from catalog_evaluation import evaluate_catalog, read_jsonl, result_metrics, write_records


class CatalogSASRec(SASRec):
    def score_catalog(self, seqs):
        return self.log2feats(seqs)[:, -1, :] @ self.item_emb.weight[1:].T


def sample_sequences(users, indices, maxlen, n_items, rng):
    seq = np.zeros((len(indices), maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    for b, i in enumerate(indices):
        items = users[i]["items"]
        length = min(maxlen, len(items) - 1)
        if length <= 0:
            continue
        seq[b, -length:] = items[-length - 1:-1]
        pos[b, -length:] = items[-length:]
        excluded = set(items)
        values = rng.integers(1, n_items + 1, size=length)
        bad = np.asarray([v in excluded for v in values])
        while bad.any():
            values[bad] = rng.integers(1, n_items + 1, size=int(bad.sum()))
            bad = np.asarray([v in excluded for v in values])
        neg[b, -length:] = values
    return seq, pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ml1m_sasrec_full.yaml")
    ap.add_argument("--data", default="data/ml1m")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="Evaluate an existing model.pt on the current valid/test files")
    args = ap.parse_args()
    cfg = read_config(ROOT / args.config)
    mc = cfg["model"]
    if mc["revision"] != "b253407b3c5d6ec3201a39767a572f44fd5ef6d0":
        raise ValueError("Config does not identify the vendored SASRec revision")
    torch.set_num_threads(mc["torch_threads"])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_per_process_memory_fraction(mc["gpu_memory_limit_gib"] * 1024**3 /
                                                  torch.cuda.get_device_properties(0).total_memory)
    device = mc["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required by this configuration")
    data = ROOT / args.data
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run_manifest.json").exists() and not args.resume and not args.eval_only:
        raise SystemExit("Existing run: use --resume rather than overwrite")
    users = read_jsonl(data / "train.jsonl")
    valid_rows = read_jsonl(data / "valid.jsonl")
    n_items = len(json.loads((data / "items.json").read_text(encoding="utf-8")))
    model_args = SimpleNamespace(**mc)
    model = CatalogSASRec(len(users), n_items, model_args).to(device)
    for _, param in model.named_parameters():
        if param.ndim > 1:
            torch.nn.init.xavier_normal_(param.data)
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    if args.eval_only:
        checkpoint = out / "model.pt"
        if not checkpoint.exists():
            raise SystemExit(f"missing checkpoint: {checkpoint}")
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        for split in ("valid", "test"):
            rows = valid_rows if split == "valid" else read_jsonl(data / "test.jsonl")
            records = evaluate_catalog(model, rows, mc["maxlen"], batch_size=mc["eval_batch_size"],
                                       exclude_seen=mc["exclude_seen"])
            write_records(out / f"{split}_candidates.jsonl", records)
            write_json(out / f"{split}_metrics.json", result_metrics(records))
        print(json.dumps({"status": "eval_complete", "out": args.out}), flush=True)
        return
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
        log, best, best_epoch, stale = last["log"], last["best"], last["best_epoch"], last["stale"]
        start_epoch = last["epoch"] + 1
    manifest = {"method": "SASRec-pytorch", "seed": args.seed, "config": cfg,
                "revision": mc["revision"], "model_source_sha256": sha256(ROOT / "third_party/sasrec_pytorch/model.py"),
                "dataset_hashes": json.loads((data / "hashes.json").read_text(encoding="utf-8")),
                "selection": "full validation NDCG@10", "test_used_for_selection": False,
                "status": "running", "started_at": time.time()}
    write_json(out / "run_manifest.json", manifest)
    stopping_reason = "max_epochs"
    for epoch in range(start_epoch, mc["max_epochs"] + 1):
        started = time.perf_counter()
        model.train()
        losses = []
        order = rng.permutation(len(users))
        for s in range(0, len(users), mc["batch_size"]):
            idx = order[s:s + mc["batch_size"]]
            seq, pos, neg = sample_sequences(users, idx, mc["maxlen"], n_items, rng)
            positive, negative = model(None, seq, pos, neg)
            keep = torch.as_tensor(pos > 0, device=device)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(positive[keep], torch.ones_like(positive[keep])) +
                    torch.nn.functional.binary_cross_entropy_with_logits(negative[keep], torch.zeros_like(negative[keep])))
            if mc["l2_emb"]:
                loss += mc["l2_emb"] * (model.item_emb.weight**2).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": time.perf_counter() - started}
        if epoch % mc["eval_every"] == 0 or epoch == mc["max_epochs"]:
            metrics = result_metrics(evaluate_catalog(model, valid_rows, mc["maxlen"],
                batch_size=mc["eval_batch_size"], exclude_seen=mc["exclude_seen"]))
            entry["valid"] = metrics
            score = metrics["all"]["ndcg@10"]
            if score > best:
                best, best_epoch, stale = score, epoch, 0
                torch.save({"model": model.state_dict(), "config": cfg, "seed": args.seed,
                            "best_epoch": epoch, "best_valid_ndcg10": best}, out / "model.pt")
            else:
                stale += 1
            log.append(entry)
            write_json(out / "train_log.json", {"best_epoch": best_epoch, "best_valid_ndcg10": best, "history": log})
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg,
                        "seed": args.seed, "epoch": epoch, "log": log, "best": best, "best_epoch": best_epoch,
                        "stale": stale, "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None}, out / "last.pt")
            print(json.dumps({"epoch": epoch, "loss": entry["loss"], "valid_ndcg10": score,
                              "candidate_recall200": metrics["all"]["candidate_recall@200"], "stale": stale}), flush=True)
            if epoch >= mc["min_epochs"] and stale >= mc["patience_evaluations"]:
                stopping_reason = "validation_patience"
                break
        else:
            log.append(entry)
    best_state = torch.load(out / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    for split in ("valid", "test"):
        rows = valid_rows if split == "valid" else read_jsonl(data / "test.jsonl")
        records = evaluate_catalog(model, rows, mc["maxlen"], batch_size=mc["eval_batch_size"],
                                   exclude_seen=mc["exclude_seen"])
        write_records(out / f"{split}_candidates.jsonl", records)
        write_json(out / f"{split}_metrics.json", result_metrics(records))
    manifest.update({"status": "complete", "best_epoch": best_epoch,
                     "epochs_ran": log[-1]["epoch"], "stopping_reason": stopping_reason,
                     "finished_at": time.time(), "checkpoint_sha256": sha256(out / "model.pt")})
    write_json(out / "run_manifest.json", manifest)
    print(json.dumps({"status": "complete", "best_epoch": best_epoch, "out": args.out}), flush=True)


if __name__ == "__main__":
    main()
