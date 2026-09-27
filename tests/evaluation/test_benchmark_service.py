import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation"))

from benchmark_service import main, multipart_image, parse_response  # noqa: E402


class ResponseParsingTests(unittest.TestCase):
    def test_flat_official_response(self):
        self.assertEqual(parse_response(b'{"slug":"wine-a"}'), ("wine-a", ["wine-a"]))

    def test_array_response_preserves_ranking(self):
        body = json.dumps([{"slug": "wine-a"}, {"slug": "wine-b"}]).encode()
        self.assertEqual(parse_response(body), ("wine-a", ["wine-a", "wine-b"]))

    def test_invalid_response(self):
        self.assertEqual(parse_response(b"not-json"), (None, None))

    def test_runner_posts_multipart_and_writes_official_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            image = images / "photo.jpg"
            image.write_bytes(b"fake image")
            body, boundary = multipart_image(image)
            self.assertTrue(body.startswith(f"--{boundary}\r\n".encode()))
            self.assertIn(b'name="image"', body)
            manifest = root / "queries.tsv"
            manifest.write_text("query_id\timage_path\nq1\tphoto.jpg\n", encoding="utf-8")
            output = root / "predictions.jsonl"
            with mock.patch(
                "benchmark_service.request_prediction",
                return_value=("wine-a", ["wine-a"], 200, None, 12.4),
            ):
                result = main([
                    "--images-dir", str(images),
                    "--manifest", str(manifest),
                    "--output", str(output),
                ])
            self.assertEqual(result, 0)
            prediction = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                set(prediction),
                {"query_id", "image_path", "image_sha256", "predicted_slug", "latency_ms"},
            )
            self.assertEqual(prediction["predicted_slug"], "wine-a")


if __name__ == "__main__":
    unittest.main()
