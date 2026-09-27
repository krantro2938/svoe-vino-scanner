import hashlib
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "evaluation"))

from evaluate import _candidate_slugs, evaluate, markdown_report  # noqa: E402


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class EvaluationTests(unittest.TestCase):
    def test_table_candidate_formats(self):
        self.assertEqual(_candidate_slugs({"top5": "a|b|c"}), ["a", "b", "c"])
        self.assertEqual(_candidate_slugs({"candidates": '[{"slug":"a"},"b"]'}), ["a", "b"])

    def test_scored_metrics_and_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "queries.tsv").write_text(
                "query_id\timage_path\texpected_slug\nq1\ta.jpg\ta\nq2\tb.jpg\tb\nq3\tc.jpg\tb\n",
                encoding="utf-8",
            )
            (root / "catalog.csv").write_text("Slug\na\nb\nc\n", encoding="utf-8")
            rows = []
            for query_id, image, slug, candidates, latency in (
                ("q1", "a.jpg", "a", ["a", "b"], 10),
                ("q2", "b.jpg", "a", ["a", "b"], 20),
                ("q3", "c.jpg", "b", ["b", "a"], 30),
            ):
                rows.append({
                    "query_id": query_id,
                    "image_path": image,
                    "image_sha256": "0" * 64,
                    "predicted_slug": slug,
                    "latency_ms": latency,
                    "candidate_slugs": candidates,
                })
            write_jsonl(root / "predictions.jsonl", rows)
            report = evaluate(root / "queries.tsv", root / "predictions.jsonl", catalog_path=root / "catalog.csv")
            self.assertEqual(report["status"], "valid")
            self.assertEqual(report["mode"], "scored")
            self.assertEqual(report["metrics"]["accuracy_at_1"], 0.667)
            self.assertEqual(report["metrics"]["micro_f1"], 0.667)
            self.assertEqual(report["metrics"]["macro_f1"], 0.667)
            self.assertEqual(report["metrics"]["recall_at_5"], 1.0)
            self.assertEqual(report["latency_ms"], {"count": 3, "p50": 20.0, "p95": 29.0, "max": 30.0})
            self.assertEqual(report["validation"]["warning_count"], 1)

    def test_grouped_hard_negative_and_split_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "queries.jsonl").write_text("".join(
                json.dumps(row) + "\n" for row in (
                    {"query_id": "q1", "image_path": "1.jpg", "expected_slug": "a", "hard_negative_group": "family", "split": "validation"},
                    {"query_id": "q2", "image_path": "2.jpg", "expected_slug": "b", "hard_negative_group": "family", "split": "validation"},
                    {"query_id": "q3", "image_path": "3.jpg", "expected_slug": "c", "hard_negative_group": "solo", "split": "test"},
                )
            ), encoding="utf-8")
            write_jsonl(root / "predictions.jsonl", [
                {"query_id": "q1", "image_path": "1.jpg", "image_sha256": "0" * 64, "predicted_slug": "a", "candidate_slugs": ["a", "b"], "latency_ms": 10},
                {"query_id": "q2", "image_path": "2.jpg", "image_sha256": "0" * 64, "predicted_slug": "a", "candidate_slugs": ["a", "b"], "latency_ms": 20},
                {"query_id": "q3", "image_path": "3.jpg", "image_sha256": "0" * 64, "predicted_slug": "c", "candidate_slugs": ["c"], "latency_ms": 30},
            ])
            report = evaluate(root / "queries.jsonl", root / "predictions.jsonl")
            grouped = report["metrics"]["grouped"]
            self.assertEqual(grouped["field"], "hard_negative_group")
            self.assertEqual(grouped["group_count"], 2)
            self.assertEqual(grouped["hard_negative"]["queries"], 2)
            self.assertEqual(grouped["hard_negative"]["accuracy_at_1"], 0.5)
            self.assertEqual(grouped["hard_negative"]["recall_at_5"], 1.0)
            self.assertEqual(report["metrics"]["by_split"]["validation"]["accuracy_at_1"], 0.5)
            self.assertEqual(report["metrics"]["by_split"]["test"]["accuracy_at_1"], 1.0)
            rendered = markdown_report(report)
            self.assertIn("Hard-negative slice: 2 queries", rendered)

    def test_macro_f1_includes_a_predicted_only_class(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "queries.tsv").write_text(
                "query_id\timage_path\texpected_slug\nq1\ta.jpg\ta\nq2\tb.jpg\tb\n",
                encoding="utf-8",
            )
            write_jsonl(root / "predictions.jsonl", [
                {"query_id": "q1", "image_path": "a.jpg", "image_sha256": "0" * 64, "predicted_slug": "a", "latency_ms": 1},
                {"query_id": "q2", "image_path": "b.jpg", "image_sha256": "0" * 64, "predicted_slug": "not-in-truth", "latency_ms": 1},
            ])
            report = evaluate(root / "queries.tsv", root / "predictions.jsonl")
            self.assertEqual(report["metrics"]["accuracy_at_1"], 0.5)
            self.assertEqual(report["metrics"]["micro_f1"], 0.5)
            self.assertEqual(report["metrics"]["macro_f1"], 0.333)
            self.assertEqual(report["metrics"]["per_class"]["not-in-truth"]["support"], 0)

    def test_unlabeled_integrity_and_real_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            image = images / "photo.jpg"
            image.write_bytes(b"image bytes")
            digest = hashlib.sha256(b"image bytes").hexdigest()
            (root / "queries.tsv").write_text("query_id\timage_path\nq1\tphoto.jpg\n", encoding="utf-8")
            (root / "catalog.jsonl").write_text('{"slug":"wine"}\n', encoding="utf-8")
            (root / "checksums.sha256").write_text(f"{digest}  queries/photo.jpg\n", encoding="utf-8")
            write_jsonl(root / "predictions.jsonl", [{
                "query_id": "q1",
                "image_path": "photo.jpg",
                "image_sha256": digest,
                "predicted_slug": "wine",
                "latency_ms": 4.5,
            }])
            report = evaluate(
                root / "queries.tsv",
                root / "predictions.jsonl",
                catalog_path=root / "catalog.jsonl",
                images_dir=images,
                checksums_path=root / "checksums.sha256",
            )
            self.assertEqual(report["status"], "valid")
            self.assertEqual(report["mode"], "unlabeled_validation")
            self.assertIsNone(report["metrics"])
            self.assertEqual(report["validation"]["checksums_matched"], 1)
            self.assertIn("integrity validation only", markdown_report(report))

    def test_unknown_slug_and_checksum_mismatch_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            (images / "x.jpg").write_bytes(b"x")
            (root / "queries.tsv").write_text("query_id\timage_path\nq1\tx.jpg\n", encoding="utf-8")
            (root / "catalog.csv").write_text("slug\nknown\n", encoding="utf-8")
            write_jsonl(root / "predictions.jsonl", [{
                "query_id": "q1",
                "image_path": "x.jpg",
                "image_sha256": "0" * 64,
                "predicted_slug": "unknown",
                "latency_ms": 1,
            }])
            report = evaluate(root / "queries.tsv", root / "predictions.jsonl", catalog_path=root / "catalog.csv", images_dir=images)
            self.assertEqual(report["status"], "invalid")
            self.assertEqual({issue["code"] for issue in report["issues"] if issue["level"] == "error"}, {"unknown_predicted_slug", "prediction_checksum_mismatch"})

    def test_missing_required_prediction_field_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "queries.tsv").write_text("query_id\timage_path\nq1\tx.jpg\n", encoding="utf-8")
            write_jsonl(root / "predictions.jsonl", [{
                "query_id": "q1",
                "image_path": "x.jpg",
                "image_sha256": "0" * 64,
                "latency_ms": 1,
            }])
            report = evaluate(root / "queries.tsv", root / "predictions.jsonl")
            self.assertEqual(report["status"], "invalid")
            self.assertIn("missing_prediction_field", {issue["code"] for issue in report["issues"]})

    def test_missing_prediction_counts_as_wrong(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "queries.tsv").write_text(
                "query_id\timage_path\texpected_slug\nq1\ta.jpg\ta\nq2\tb.jpg\tb\n",
                encoding="utf-8",
            )
            write_jsonl(root / "predictions.jsonl", [{
                "query_id": "q1",
                "image_path": "a.jpg",
                "image_sha256": "0" * 64,
                "predicted_slug": "a",
                "latency_ms": 1,
            }])
            report = evaluate(root / "queries.tsv", root / "predictions.jsonl")
            self.assertEqual(report["metrics"]["labeled_queries"], 2)
            self.assertEqual(report["metrics"]["accuracy_at_1"], 0.5)


if __name__ == "__main__":
    unittest.main()
