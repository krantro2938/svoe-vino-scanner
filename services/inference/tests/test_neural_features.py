from __future__ import annotations

import json

import numpy as np
from PIL import Image

from app.config import Settings
from app.imaging import DecodedImage
from app.neural_features import NeuralFeatures
from app.retrieval import RecognitionEngine


class FakeSession:
    def __init__(self, embedding):
        self.embedding = np.asarray([embedding], dtype=np.float32)

    def run(self, _outputs, inputs):
        assert inputs["pixel_values"].shape == (1, 3, 224, 224)
        return [self.embedding]


def stub_neural():
    neural = NeuralFeatures.__new__(NeuralFeatures)
    neural.manifest = {"center_width": 0.7, "input_size": 224}
    neural.embeddings = np.asarray(((1.0, 0.0), (0.9, 0.1), (0.0, 1.0)), dtype=np.float32)
    neural.slugs = ["a", "a", "b"]
    neural.session = FakeSession((1.0, 0.0))
    return neural


def test_neural_preprocessing_and_duplicate_label_collapse():
    result = stub_neural().predict(Image.new("RGB", (640, 640), "red"))
    assert result["slug"] == "a"
    assert result["top5"] == ["a", "b"]
    assert result["margin"] > 0.9


def test_neural_fallback_replaces_handcrafted_visual_before_ocr(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps([{"slug": "a", "name": "A"}, {"slug": "b", "name": "B"}]))
    engine = RecognitionEngine(Settings(catalog_path=catalog, index_manifest_path=None))
    engine._neural_features = stub_neural()
    decoded = DecodedImage(Image.new("RGB", (64, 96), "red"), "PNG", 64, 96)
    match = engine.predict(decoded)
    assert match.wine.slug == "a"
    assert match.method == "dinov2"


def test_wide_scene_with_side_objects_keeps_center_visual_fallback(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps([{"slug": "a", "name": "A"}, {"slug": "b", "name": "B"}]))
    engine = RecognitionEngine(Settings(catalog_path=catalog, index_manifest_path=None))
    neural = stub_neural()
    neural.slugs = ["b", "b", "b"]
    engine._neural_features = neural
    image = Image.new("RGB", (384, 160), "white")
    image.paste("black", (20, 0, 84, 160))
    image.paste("black", (160, 0, 224, 160))
    image.paste("black", (300, 0, 364, 160))
    decoded = DecodedImage(image, "PNG", 384, 160)
    match = engine.predict(decoded)
    assert match.wine.slug == "a"
    assert match.method == "visual"
