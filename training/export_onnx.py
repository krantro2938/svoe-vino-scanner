#!/usr/bin/env python3
"""Export the selected DINOv2 projection and its gallery for CPU inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from embedding_experiment import MODEL, REVISION, records


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slug_fingerprint(slugs: list[str]) -> str:
    payload = json.dumps(sorted(set(slugs)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class RetrievalEmbedding(nn.Module):
    def __init__(self, backbone: nn.Module, projection: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.projection = projection

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(pixel_values=pixel_values).last_hidden_state
        embedding = F.normalize(
            torch.cat((hidden[:, 0], hidden[:, 1:].mean(dim=1)), dim=1), dim=1
        )
        return F.normalize(self.projection(embedding), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("training/artifacts/dinov2-small"))
    parser.add_argument("--references", type=Path, default=Path("data/generated/expanded/training_references.jsonl"))
    parser.add_argument("--catalog", type=Path, default=Path("data/generated/expanded/catalog.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/generated/expanded/dinov2"))
    args = parser.parse_args()

    report = json.loads((args.experiment / "report.json").read_text())
    if report["model"] != MODEL or report["revision"] != REVISION:
        raise RuntimeError("Experiment model revision does not match exporter")
    if not report["selected"].startswith("projection_epoch_"):
        raise RuntimeError("Experiment did not select a trained projection")
    references = records(args.references)
    catalog = records(args.catalog)
    catalog_slugs = [row["slug"] for row in catalog]
    if any(row["slug"] not in set(catalog_slugs) for row in references):
        raise RuntimeError("Reference slug is absent from catalogue")

    checkpoint_path = args.experiment / "projection.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    backbone = AutoModel.from_pretrained(
        MODEL,
        revision=REVISION,
        cache_dir=str(args.experiment / "model"),
        use_safetensors=True,
        local_files_only=True,
    ).eval()
    dimension = backbone.config.hidden_size * 2
    projection = nn.Linear(dimension, dimension, bias=False)
    projection.load_state_dict(checkpoint["state_dict"])
    wrapper = RetrievalEmbedding(backbone, projection).eval()

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "model.onnx"
    torch.onnx.export(
        wrapper,
        (torch.zeros(1, 3, 224, 224),),
        model_path,
        input_names=["pixel_values"],
        output_names=["embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(model_path))

    cache = np.load(args.experiment / "features.npz")
    gallery = cache["gallery"].astype(np.float32)
    cache.close()
    weight = projection.weight.detach().numpy().astype(np.float32)
    gallery = gallery @ weight.T
    gallery /= np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-8)
    if len(gallery) != len(references):
        raise RuntimeError("Gallery/reference length mismatch")
    gallery_path = args.output / "gallery.npz"
    np.savez_compressed(
        gallery_path,
        embeddings=gallery,
        slugs=np.asarray([row["slug"] for row in references]),
    )
    metadata = {
        "schema_version": 1,
        "model": MODEL,
        "revision": REVISION,
        "selected": report["selected"],
        "preprocessing_version": 1,
        "center_width": 0.70,
        "input_size": 224,
        "embedding_dimension": int(gallery.shape[1]),
        "reference_count": len(references),
        "references_sha256": sha256(args.references),
        "catalog_slug_fingerprint": slug_fingerprint(catalog_slugs),
        "projection_sha256": sha256(checkpoint_path),
        "model_sha256": sha256(model_path),
        "gallery_sha256": sha256(gallery_path),
        "validation_accuracy_at_1": report["selected_validation_accuracy_at_1"],
        "test_queries_used": report["test_queries_used"],
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
