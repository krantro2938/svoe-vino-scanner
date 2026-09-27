#!/usr/bin/env python3
"""Frozen DINOv2 retrieval and train-only identity-initialized metric learning; no test queries."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

MODEL = "facebook/dinov2-small"
REVISION = "150c8e7bb7cef2d30ec31b13a517af14840ee3f7"


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preprocess(path):
    with Image.open(path) as opened:
        rgba = ImageOps.exif_transpose(opened).convert("RGBA")
        image = Image.new("RGB", rgba.size, "white")
        image.paste(rgba, mask=rgba.getchannel("A"))
    width, height = image.size
    if width / height > .85:
        crop_width = round(height * .70)
        left = (width - crop_width) // 2
        image = image.crop((left, 0, left + crop_width, height))
    # Standard DINOv2 256-short-edge resize then 224 centre crop.
    scale = 256 / min(image.size)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BICUBIC)
    left, top = (image.width - 224) // 2, (image.height - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))
    array = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255
    return (torch.from_numpy(array) - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


@torch.inference_mode()
def encode(model, paths, batch_size):
    values = []
    started = time.time()
    for offset in range(0, len(paths), batch_size):
        batch = torch.stack([preprocess(path) for path in paths[offset:offset + batch_size]])
        hidden = model(pixel_values=batch).last_hidden_state
        embedding = F.normalize(torch.cat((hidden[:, 0], hidden[:, 1:].mean(dim=1)), dim=1), dim=1)
        values.append(embedding.numpy())
        if offset == 0 or (offset // batch_size) % 20 == 0:
            print(json.dumps({"encoded": min(offset + batch_size, len(paths)), "total": len(paths), "elapsed_seconds": round(time.time() - started, 1)}), flush=True)
    return np.concatenate(values)


def measure(query, gallery, query_rows, gallery_rows):
    q = F.normalize(torch.as_tensor(query), dim=1)
    g = F.normalize(torch.as_tensor(gallery), dim=1)
    similarities = (q @ g.T).numpy()
    labels = [row["slug"] for row in gallery_rows]
    correct = 0
    top5 = 0
    predictions = []
    scenes = {}
    for row, scores in zip(query_rows, similarities):
        order = np.argsort(-scores)
        unique = []
        for index in order:
            if labels[index] not in unique:
                unique.append(labels[index])
            if len(unique) == 5:
                break
        hit = unique[0] == row["expected_slug"]
        correct += hit
        top5 += row["expected_slug"] in unique
        scene = scenes.setdefault(row["scene_type"], {"correct": 0, "count": 0})
        scene["correct"] += hit
        scene["count"] += 1
        predictions.append({"query_id": row["query_id"], "expected_slug": row["expected_slug"], "predicted_slug": unique[0], "top5": unique})
    return {"count": len(query_rows), "accuracy_at_1": correct / len(query_rows),
            "recall_at_5": top5 / len(query_rows), "scenes": scenes}, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/artifacts/training-expanded-v3"))
    parser.add_argument("--references", type=Path, default=Path("data/generated/expanded/training_references.jsonl"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/generated/expanded/images"))
    parser.add_argument("--output", type=Path, default=Path("training/artifacts/dinov2-small"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(20260921)
    np.random.seed(20260921)
    args.output.mkdir(parents=True, exist_ok=True)
    references = records(args.references)
    all_rows = records(args.dataset / "queries.jsonl")
    # Test rows are never decoded, embedded, scored, or used to select checkpoints.
    validation = [row for row in all_rows if row["split"] == "validation"]
    training = [row for row in all_rows if row["split"] == "train" and row["variant"] in (0, 4)]
    train_slugs = {row["expected_slug"] for row in training}
    assert not train_slugs.intersection(row["expected_slug"] for row in validation)
    fingerprint = {"model": MODEL, "revision": REVISION, "references_sha256": digest(args.references),
                   "queries_sha256": digest(args.dataset / "queries.jsonl"), "preprocessing_version": 1}
    cache = args.output / "features.npz"
    metadata = args.output / "features.json"
    if cache.exists():
        if json.loads(metadata.read_text()) != fingerprint:
            raise RuntimeError("Feature cache belongs to different manifests/model/preprocessing")
        features = np.load(cache)
        gallery, val, train = features["gallery"], features["validation"], features["train"]
    else:
        model = AutoModel.from_pretrained(MODEL, revision=REVISION, cache_dir=str(args.output / "model"), use_safetensors=True).eval()
        gallery = encode(model, [args.images_dir / row["image_path"] for row in references], args.batch_size)
        val = encode(model, [args.dataset / row["image_path"] for row in validation], args.batch_size)
        baseline, predictions = measure(val, gallery, validation, references)
        (args.output / "baseline.json").write_text(json.dumps({**fingerprint, "validation": baseline}, indent=2) + "\n")
        (args.output / "baseline_predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
        print(json.dumps({"baseline_validation": baseline}), flush=True)
        train = encode(model, [args.dataset / row["image_path"] for row in training], args.batch_size)
        np.savez_compressed(cache, gallery=gallery, validation=val, train=train)
        metadata.write_text(json.dumps(fingerprint, indent=2) + "\n")
        del model
    baseline, _ = measure(val, gallery, validation, references)
    dimension = gallery.shape[1]
    projection = nn.Linear(dimension, dimension, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(dimension))
    optimizer = torch.optim.AdamW(projection.parameters(), lr=1e-4, weight_decay=.01)
    # Only training classes enter the loss, including the denominator.
    gallery_indices = [i for i, row in enumerate(references) if row["slug"] in train_slugs]
    train_gallery = torch.from_numpy(gallery[gallery_indices])
    train_gallery_labels = [references[i]["slug"] for i in gallery_indices]
    positives = torch.tensor([[row["expected_slug"] == label for label in train_gallery_labels] for row in training])
    train_features = torch.from_numpy(train)
    history = []
    best = baseline["accuracy_at_1"]
    selected = "frozen_backbone"
    for epoch in range(args.epochs):
        losses = []
        for indices in torch.randperm(len(training)).split(128):
            query = F.normalize(projection(train_features[indices]), dim=1)
            ref = F.normalize(projection(train_gallery), dim=1)
            logits = query @ ref.T / .07
            positive = logits.masked_fill(~positives[indices], -torch.inf).logsumexp(dim=1)
            loss = (logits.logsumexp(dim=1) - positive).mean()
            loss += .1 * (projection.weight - torch.eye(dimension)).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        with torch.no_grad():
            result, predictions = measure(projection(torch.from_numpy(val)), projection(torch.from_numpy(gallery)), validation, references)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation": result})
        print(json.dumps(history[-1]), flush=True)
        if result["accuracy_at_1"] > best:
            best = result["accuracy_at_1"]
            selected = f"projection_epoch_{epoch + 1}"
            torch.save({"state_dict": projection.state_dict(), "fingerprint": fingerprint, "epoch": epoch + 1}, args.output / "projection.pt")
            (args.output / "selected_predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
    report = {**fingerprint, "baseline_validation": baseline, "selected": selected,
              "selected_validation_accuracy_at_1": best, "history": history,
              "training_queries": len(training), "validation_queries": len(validation),
              "test_queries_used": 0, "gallery_references": len(references),
              "limitation": "Synthetic catalogue derivatives; not independent real-world accuracy. Backbone frozen; only projection is trained."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
