#!/usr/bin/env python3
"""Compare frozen backbones and crop policies on a labelled field benchmark.

Embeds every catalogue reference (whole bottle and label-region crops) and every
query (several centre crops), then reports Accuracy@1 and Recall@K for each
query/gallery crop pairing. Features are cached per backbone.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from field_benchmark import bottle_cutout  # noqa: E402


def square(image: Image.Image, fill=(255, 255, 255)) -> Image.Image:
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def gallery_views(path: Path) -> dict[str, Image.Image]:
    cut = bottle_cutout(path)
    flat = Image.new("RGBA", cut.size, (255, 255, 255, 255))
    flat.alpha_composite(cut)
    bottle = flat.convert("RGB")
    w, h = bottle.size
    # Label region: lower-middle part of the bottle body, roughly square.
    top = int(h * 0.42)
    label = bottle.crop((0, top, w, min(h, top + int(w * 1.3))))
    return {"bottle": square(bottle), "label": square(label)}


def query_views(image: Image.Image) -> dict[str, Image.Image]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    views = {"full": square(image, (0, 0, 0))}
    for name, fw, fh, cy in (("c80", 0.8, 0.6, 0.5), ("c60", 0.6, 0.45, 0.52)):
        cw, ch = int(w * fw), int(h * fh)
        left, top = (w - cw) // 2, int(h * cy - ch / 2)
        views[name] = square(image.crop((left, top, left + cw, top + ch)), (0, 0, 0))
    return views


class Backbone:
    def __init__(self, name: str):
        from transformers import AutoImageProcessor, AutoModel
        self.name = name
        self.model = AutoModel.from_pretrained(name).eval()
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.is_siglip = "siglip" in name

    @torch.no_grad()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        if self.is_siglip:
            features = self.model.get_image_features(**inputs)
        else:
            out = self.model(**inputs).last_hidden_state
            features = torch.cat([out[:, 0], out[:, 1:].mean(1)], dim=1)
        return torch.nn.functional.normalize(features, dim=-1).numpy()

    @torch.no_grad()
    def embed_text(self, texts: list[str]) -> np.ndarray:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.name)
        chunks = []
        for i in range(0, len(texts), 64):
            batch = tokenizer(texts[i:i + 64], padding="max_length", max_length=64, truncation=True,
                              return_tensors="pt")
            chunks.append(torch.nn.functional.normalize(self.model.get_text_features(**batch), dim=-1).numpy())
        return np.concatenate(chunks)


def batched(model: Backbone, images: list[Image.Image], size=16) -> np.ndarray:
    return np.concatenate([model.embed(images[i:i + size]) for i in range(0, len(images), size)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/siglip2-base-patch16-256")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/generated/expanded/catalog.jsonl")
    parser.add_argument("--images-dir", type=Path, default=ROOT / "data/generated/expanded/images")
    parser.add_argument("--queries", type=Path, default=ROOT / "evaluation/artifacts/field-v1/queries.jsonl")
    parser.add_argument("--cache", type=Path, default=ROOT / "training/artifacts/probe")
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()
    torch.set_num_threads(10)

    catalog = [json.loads(l) for l in args.catalog.read_text().splitlines()]
    catalog = [r for r in catalog if r.get("index_ready") and (args.images_dir / r["photo_name"]).is_file()]
    slugs = [r["slug"] for r in catalog]
    model = Backbone(args.model)
    tag = args.model.replace("/", "_")
    args.cache.mkdir(parents=True, exist_ok=True)
    gallery_file = args.cache / f"{tag}-gallery.npz"
    if gallery_file.exists():
        data = np.load(gallery_file)
        gallery = {k: data[k] for k in data.files}
    else:
        started = time.time()
        views = {"bottle": [], "label": []}
        for i, row in enumerate(catalog):
            for k, v in gallery_views(args.images_dir / row["photo_name"]).items():
                views[k].append(v)
            if (i + 1) % 500 == 0:
                print(f"gallery views {i + 1}", flush=True)
        gallery = {k: batched(model, v) for k, v in views.items()}
        np.savez(gallery_file, **gallery)
        print(f"gallery embedded in {time.time() - started:.0f}s", flush=True)

    queries = [json.loads(l) for l in args.queries.read_text().splitlines()][: args.limit]
    qviews: dict[str, list[Image.Image]] = {}
    for row in queries:
        with Image.open(args.queries.parent / row["image_path"]) as image:
            for k, v in query_views(image).items():
                qviews.setdefault(k, []).append(v)
    started = time.time()
    qemb = {k: batched(model, v) for k, v in qviews.items()}
    print(f"query embedding {1000 * (time.time() - started) / len(queries) / len(qemb):.0f} ms/view", flush=True)
    expected = np.array([slugs.index(r["expected_slug"]) for r in queries])

    def report(name: str, scores: np.ndarray) -> None:
        order = np.argsort(-scores, axis=1)
        ranks = np.array([np.flatnonzero(order[i] == expected[i])[0] for i in range(len(expected))])
        print(f"{name:28s} acc@1={np.mean(ranks < 1):.3f} R@5={np.mean(ranks < 5):.3f} "
              f"R@20={np.mean(ranks < 20):.3f} R@50={np.mean(ranks < 50):.3f}")

    for qk, qv in qemb.items():
        for gk, gv in gallery.items():
            report(f"{qk}->{gk}", qv @ gv.T)
    combined = np.max(np.stack([qv @ gv.T for qv in qemb.values() for gv in gallery.values()]), axis=0)
    report("max(all)", combined)
    mean = np.mean(np.stack([qv @ gv.T for qv in qemb.values() for gv in gallery.values()]), axis=0)
    report("mean(all)", mean)
    if model.is_siglip:
        texts = [f"{r.get('winery') or ''} {r['name']} {r.get('category') or ''}".strip() for r in catalog]
        temb = model.embed_text(texts)
        text_scores = np.max(np.stack([qv @ temb.T for qv in qemb.values()]), axis=0)
        report("image->text", text_scores)
        z = lambda s: (s - s.mean(1, keepdims=True)) / s.std(1, keepdims=True)
        report("mean + text (z)", z(mean) + 0.5 * z(text_scores))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
