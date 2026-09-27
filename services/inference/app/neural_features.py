"""Optional ONNX DINOv2 retrieval fallback selected on validation data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug_fingerprint(slugs) -> str:
    payload = json.dumps(sorted(set(slugs)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class NeuralFeatures:
    def __init__(self, directory: Path, catalog) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime is required for the DINOv2 fallback") from exc
        manifest_path = directory / "manifest.json"
        model_path = directory / "model.onnx"
        gallery_path = directory / "gallery.npz"
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("schema_version") != 1:
            raise ValueError("unsupported neural index schema")
        if self.manifest.get("test_queries_used") != 0:
            raise ValueError("neural checkpoint was selected with test-query access")
        if self.manifest.get("model_sha256") != _sha256(model_path):
            raise ValueError("neural model checksum mismatch")
        if self.manifest.get("gallery_sha256") != _sha256(gallery_path):
            raise ValueError("neural gallery checksum mismatch")
        if self.manifest.get("catalog_slug_fingerprint") != _slug_fingerprint(catalog):
            raise ValueError("neural gallery catalogue mismatch")
        data = np.load(gallery_path, allow_pickle=False)
        self.embeddings = data["embeddings"].astype(np.float32, copy=False)
        self.slugs = data["slugs"].tolist()
        data.close()
        dimension = int(self.manifest["embedding_dimension"])
        if (
            self.embeddings.shape != (len(self.slugs), dimension)
            or not np.isfinite(self.embeddings).all()
            or any(slug not in catalog for slug in self.slugs)
        ):
            raise ValueError("invalid neural gallery arrays")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        width, height = image.size
        if width / height > 0.85:
            crop_width = round(height * float(self.manifest["center_width"]))
            left = (width - crop_width) // 2
            image = image.crop((left, 0, left + crop_width, height))
        scale = 256 / min(image.size)
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.BICUBIC
        )
        size = int(self.manifest["input_size"])
        left, top = (image.width - size) // 2, (image.height - size) // 2
        image = image.crop((left, top, left + size, top + size))
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
        std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]
        return ((array - mean) / std)[None]

    def predict(self, image: Image.Image) -> dict:
        embedding = self.session.run(None, {"pixel_values": self.preprocess(image)})[0][0]
        embedding = embedding / max(float(np.linalg.norm(embedding)), 1e-8)
        scores = self.embeddings @ embedding
        by_slug: dict[str, float] = {}
        for slug, score in zip(self.slugs, scores):
            by_slug[slug] = max(by_slug.get(slug, -1.0), float(score))
        ranked = sorted(by_slug.items(), key=lambda item: (-item[1], item[0]))
        first = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0
        return {
            "slug": first[0],
            "score": first[1],
            "margin": max(0.0, first[1] - second_score),
            "top5": [slug for slug, _ in ranked[:5]],
        }
