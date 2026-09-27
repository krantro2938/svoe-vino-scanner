import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))
from audit_training_dataset import audit
from prepare_training_references import prepare
from test_synthetic_dataset import write_ppm


class TrainingAuditTests(unittest.TestCase):
    def test_references_quarantine_all_conflicting_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_ppm(root / "same.ppm", (1, 2, 3))
            write_ppm(root / "valid.ppm", (3, 2, 1))
            rows = [{"slug": slug, "photo_name": photo, "winery": "producer"}
                    for slug, photo in (("a", "same.ppm"), ("b", "same.ppm"), ("c", "valid.ppm"))]
            catalog = root / "catalog.jsonl"
            catalog.write_text("\n".join(json.dumps(row) for row in rows))
            output = root / "references.jsonl"
            result = prepare(catalog, root, output)
            self.assertEqual(result["references"], 1)
            self.assertEqual(result["rejection_counts"]["identical_bytes_conflicting_labels"], 2)
            reference = json.loads(output.read_text())
            self.assertEqual(reference["slug"], "c")
            self.assertEqual(reference["leakage_group"], "producer")

    def test_audit_detects_background_leakage_and_tampered_query(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_ppm(root / "query.ppm", (1, 2, 3))
            row = {"query_id": "one", "image_path": "query.ppm", "expected_slug": "a",
                   "source_sha256": "aaa", "split": "train", "sha256": "invalid",
                   "objects": [{"slug": "b", "source_sha256": "bbb"}]}
            other = {**row, "query_id": "two", "expected_slug": "b", "source_sha256": "bbb",
                     "split": "test", "objects": []}
            (root / "queries.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(other))
            report = audit(root)
            self.assertFalse(report["valid"])
            self.assertIn("source crosses splits: bbb", report["errors"])
            self.assertIn("class crosses splits: b", report["errors"])
            self.assertTrue(any("checksum mismatch" in error for error in report["errors"]))
