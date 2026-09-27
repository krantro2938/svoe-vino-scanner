#!/usr/bin/env python3
"""Fit a linear domain adapter on frozen SigLIP 2 embeddings (studio -> field).

Catalogue references are studio shots; queries are shelf photos. A single
768x768 matrix W (identity-initialised, applied to both sides, then L2-norm) is
trained with a softmax over the whole catalogue: the score of a slug is the mean
of its query-view x gallery-view cosine similarities, exactly as at runtime.

Training queries: cached embeddings of a field-like training set (dump cache).
Evaluation: a different field-like set, reported separately for wines unseen
during adapter training, so memorising individual wines would be visible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

QUERY_VIEWS = ("full", "centre")
GALLERY_VIEWS = ("bottle", "label")


def load_queries(cache: Path, queries: Path, slug_index: dict[str, int]):
    expected = {json.loads(l)["query_id"]: json.loads(l)["expected_slug"] for l in queries.read_text().splitlines()}
    xs, ys = [], []
    for line in cache.read_text().splitlines():
        row = json.loads(line)
        slug = expected.get(row["query_id"])
        if slug in slug_index and "emb" in row:
            xs.append(np.stack([np.asarray(row["emb"][v], dtype=np.float32) for v in QUERY_VIEWS]))
            ys.append(slug_index[slug])
    return torch.tensor(np.stack(xs)), torch.tensor(ys)


def scores(W: torch.Tensor, queries: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
    q = F.normalize(queries @ W, dim=-1)          # (n, Vq, d)
    g = F.normalize(gallery @ W, dim=-1)          # (Vg, m, d)
    return torch.einsum("nqd,gmd->nm", q, g) / (q.shape[1] * g.shape[0])


def accuracy(W, queries, labels, gallery, mask=None):
    with torch.no_grad():
        s = scores(W, queries, gallery)
        rank = (s > s.gather(1, labels[:, None])).sum(1)
        if mask is not None:
            rank = rank[mask]
        return float((rank < 1).float().mean()), float((rank < 5).float().mean()), float((rank < 30).float().mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", type=Path, required=True, help="siglip gallery.npz")
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--train-queries", type=Path, required=True)
    parser.add_argument("--eval-cache", type=Path, required=True)
    parser.add_argument("--eval-queries", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--l2", type=float, default=1.0, help="pull towards identity")
    args = parser.parse_args()
    torch.manual_seed(20260927)

    data = np.load(args.gallery)
    slugs = data["slugs"].tolist()
    slug_index = {s: i for i, s in enumerate(slugs)}
    gallery = torch.tensor(np.stack([data[v].astype(np.float32) for v in GALLERY_VIEWS]))
    xq, yq = load_queries(args.train_cache, args.train_queries, slug_index)
    # Model selection uses wines held out of the training set by slug hash; the
    # separate evaluation set is only reported, never used to choose epochs.
    held = torch.tensor([int(hashlib.sha256(slugs[int(y)].encode()).hexdigest()[:8], 16) % 5 == 0 for y in yq])
    xv, yv = xq[held], yq[held]
    xq, yq = xq[~held], yq[~held]
    xe, ye = load_queries(args.eval_cache, args.eval_queries, slug_index)
    unseen = torch.tensor([int(y) not in set(yq.tolist()) for y in ye])
    dim = gallery.shape[-1]
    identity = torch.eye(dim)
    print(f"train queries {len(yq)}, selection {len(yv)}, eval {len(ye)} ({int(unseen.sum())} wines unseen in training)")
    print("zero-shot       eval acc@1/R@5/R@30 = %.3f/%.3f/%.3f" % accuracy(identity, xe, ye, gallery),
          " unseen: %.3f/%.3f/%.3f" % accuracy(identity, xe, ye, gallery, unseen))

    W = torch.nn.Parameter(identity.clone())
    log_scale = torch.nn.Parameter(torch.tensor(np.log(30.0), dtype=torch.float32))
    optimiser = torch.optim.Adam([W, log_scale], lr=1e-3)
    best = (accuracy(identity, xv, yv, gallery)[0], identity.clone(), 0)
    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(len(yq))
        for start in range(0, len(yq), 128):
            batch = permutation[start:start + 128]
            logits = scores(W, xq[batch], gallery) * log_scale.exp()
            loss = F.cross_entropy(logits, yq[batch]) + args.l2 * ((W - identity) ** 2).sum() / dim
            optimiser.zero_grad(); loss.backward(); optimiser.step()
        if epoch % 10 == 0:
            ev = accuracy(W.detach(), xe, ye, gallery)
            un = accuracy(W.detach(), xe, ye, gallery, unseen)
            tr = accuracy(W.detach(), xq, yq, gallery)
            va = accuracy(W.detach(), xv, yv, gallery)
            print(f"epoch {epoch:3d} loss {loss.item():.3f} train {tr[0]:.3f} | select {va[0]:.3f} | "
                  f"eval {ev[0]:.3f}/{ev[1]:.3f}/{ev[2]:.3f} | unseen {un[0]:.3f}/{un[1]:.3f}/{un[2]:.3f}")
            if va[0] > best[0]:
                best = (va[0], W.detach().clone(), epoch)
    if args.output:
        matrix = best[1].numpy().astype(np.float32)
        np.save(args.output, matrix)
        digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
        args.output.with_suffix(".json").write_text(json.dumps({
            "adapter": "linear-v1", "sha256": digest, "selection_acc1": round(best[0], 4), "epoch": best[2],
            "train_queries": str(args.train_queries), "l2_to_identity": args.l2,
        }, indent=2))
        print(f"saved {args.output} (epoch {best[2]}, selection acc@1 {best[0]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
