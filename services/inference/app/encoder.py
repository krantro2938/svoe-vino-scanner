"""SigLIP 2 image-embedding retrieval over multi-view catalogue galleries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .catalog import SCENE_PHOTO_SLUGS
from .views import GALLERY_VIEWS, QUERY_VIEWS, query_views


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug_fingerprint(slugs) -> str:
    payload = json.dumps(sorted(set(slugs)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class ImageEncoder:
    def __init__(self, directory: Path, threads: int = 6) -> None:
        import onnxruntime as ort

        self.config = json.loads((directory / "encoder.json").read_text())
        model = directory / "model.onnx"
        if self.config.get("model_sha256") != _sha256(model):
            raise ValueError("encoder checksum mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.size = int(self.config["input_size"])
        self.mean = np.asarray(self.config["mean"], dtype=np.float32)[:, None, None]
        self.std = np.asarray(self.config["std"], dtype=np.float32)[:, None, None]

    def preprocess(self, image: Image.Image) -> np.ndarray:
        resized = image.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return (array - self.mean) / self.std

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        batch = np.stack([self.preprocess(image) for image in images])
        return self.session.run(None, {"pixel_values": batch})[0].astype(np.float32)


class VisualIndex:
    """Scores every catalogue slug by view-to-view cosine similarity."""

    def __init__(self, directory: Path, catalog: dict, threads: int = 6) -> None:
        self.encoder = ImageEncoder(directory, threads)
        gallery_path = directory / "gallery.npz"
        manifest = json.loads((directory / "gallery.json").read_text())
        if manifest.get("gallery_sha256") != _sha256(gallery_path):
            raise ValueError("gallery checksum mismatch")
        if manifest.get("model_sha256") != self.encoder.config["model_sha256"]:
            raise ValueError("gallery was built with a different encoder")
        data = np.load(gallery_path, allow_pickle=False)
        self.slugs = data["slugs"].tolist()
        if any(slug not in catalog for slug in self.slugs):
            raise ValueError("gallery contains slugs absent from the catalogue")
        self.views = {name: data[name].astype(np.float32) for name in GALLERY_VIEWS}
        data.close()
        # Optional studio->field linear adapter (training/fit_adapter.py).
        self.adapter = None
        adapter_path = directory / "adapter.npy"
        if adapter_path.is_file():
            meta = json.loads(adapter_path.with_suffix(".json").read_text())
            if meta.get("sha256") != _sha256(adapter_path):
                raise ValueError("adapter checksum mismatch")
            self.adapter = np.load(adapter_path).astype(np.float32)
            self.views = {name: self._project(v) for name, v in self.views.items()}
        self.index_of = {slug: i for i, slug in enumerate(self.slugs)}
        self.unreliable = np.array([self.index_of[s] for s in SCENE_PHOTO_SLUGS if s in self.index_of], dtype=int)

    def _project(self, vectors: np.ndarray) -> np.ndarray:
        if self.adapter is None:
            return vectors
        projected = vectors @ self.adapter
        return projected / np.maximum(np.linalg.norm(projected, axis=-1, keepdims=True), 1e-8)

    def embed_query(self, image: Image.Image) -> dict[str, np.ndarray]:
        """Raw encoder output per query view (adapter is applied in scores)."""
        views = query_views(image)
        vectors = self.encoder.encode([views[name] for name in QUERY_VIEWS])
        return dict(zip(QUERY_VIEWS, vectors))

    def scores(self, query: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Per view-pair similarities, plus their max and mean, per slug."""
        pairs = {f"{q}>{g}": self.views[g] @ self._project(vector) for q, vector in query.items() for g in GALLERY_VIEWS}
        stacked = np.stack(list(pairs.values()))
        pairs["max"] = stacked.max(axis=0)
        pairs["mean"] = stacked.mean(axis=0)
        for scores in pairs.values():   # no usable reference photo: never a visual match
            scores[self.unreliable] = scores.min()
        return pairs
