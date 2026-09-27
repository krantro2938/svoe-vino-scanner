#!/usr/bin/env python3
"""Export the SigLIP 2 image tower (optionally fine-tuned) to ONNX.

The exported graph maps normalised 256x256 RGB pixels to an L2-normalised image
embedding. The gallery is built separately by the inference service
(`services/inference/tools/build_siglip_index.py`) so indexing and queries share
one preprocessing implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

MODEL = "google/siglip2-base-patch16-256"


class ImageTower(nn.Module):
    def __init__(self, model: nn.Module, projection: nn.Module | None):
        super().__init__()
        self.vision = model.vision_model
        self.projection = projection

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = self.vision(pixel_values=pixel_values).pooler_output
        if self.projection is not None:
            pooled = self.projection(pooled)
        return F.normalize(pooled, dim=-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--checkpoint", type=Path, help="fine-tuned state dict from finetune_siglip.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model).eval()
    projection = None
    source = "zero-shot"
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.vision_model.load_state_dict(state["vision"])
        if state.get("projection") is not None:
            weight = state["projection"]["weight"]
            projection = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
            projection.load_state_dict(state["projection"])
        source = f"fine-tuned:{args.checkpoint.name}"
    tower = ImageTower(model, projection).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output / "model.onnx"
    dummy = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        reference = tower(dummy).numpy()
        torch.onnx.export(
            tower, (dummy,), str(onnx_path), input_names=["pixel_values"], output_names=["embedding"],
            dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17,
            dynamo=False,
        )
    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    (args.output / "encoder.json").write_text(json.dumps({
        "schema_version": 2,
        "backbone": args.model,
        "weights": source,
        "input_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "embedding_dimension": int(reference.shape[1]),
        "model_sha256": digest,
        "reference_embedding_first8": np.round(reference[0, :8], 5).tolist(),
    }, indent=2))
    print(json.dumps({"onnx": str(onnx_path), "sha256": digest, "dim": int(reference.shape[1])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
