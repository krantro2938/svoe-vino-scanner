import unittest
import numpy as np
try:
    from embedding_experiment import measure
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise unittest.SkipTest("training-only torch environment is not installed") from exc


class RetrievalMetricsTests(unittest.TestCase):
    def test_duplicate_gallery_photos_do_not_consume_top5_label_slots(self):
        gallery = np.array([[1., 0.], [.99, .01], [0., 1.], [-1., 0.]], dtype=np.float32)
        query = np.array([[1., 0.]], dtype=np.float32)
        rows = [{"query_id": "q", "expected_slug": "a", "scene_type": "single_reference"}]
        metrics, predictions = measure(query, gallery, rows, [{"slug": slug} for slug in ("a", "a", "b", "c")])
        self.assertEqual(metrics["accuracy_at_1"], 1.)
        self.assertEqual(metrics["recall_at_5"], 1.)
        self.assertEqual(predictions[0]["top5"], ["a", "b", "c"])

    def test_metrics_count_wrong_top1_and_preserve_scene_counts(self):
        gallery = np.eye(2, dtype=np.float32)
        rows = [{"query_id": "q", "expected_slug": "b", "scene_type": "center_target_three_bottles"}]
        metrics, _ = measure(gallery[:1], gallery, rows, [{"slug": "a"}, {"slug": "b"}])
        self.assertEqual(metrics["accuracy_at_1"], 0.)
        self.assertEqual(metrics["recall_at_5"], 1.)
        self.assertEqual(metrics["scenes"]["center_target_three_bottles"], {"correct": 0, "count": 1})


if __name__ == "__main__":
    unittest.main()
