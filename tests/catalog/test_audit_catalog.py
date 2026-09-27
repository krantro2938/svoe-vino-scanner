import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_catalog import (
    audit_catalog,
    load_jsonl,
    load_label_aliases,
    unmatched_query_tokens,
)


def record(slug, photo, name, winery, status="resolved", blockers=None):
    blockers = blockers or []
    value = {
        "slug": slug,
        "photo_name": photo,
        "name": name,
        "winery": winery,
        "category": "Красное",
        "grapes": "Пино Нуар",
        "region": "Крым",
        "media_status": status,
        "index_ready": not blockers,
    }
    if blockers:
        value["index_blockers"] = blockers
    return value


class CatalogAuditTests(unittest.TestCase):
    def test_recomputes_blockers_and_expands_shared_photo_conflict(self):
        rows = [
            record("a", "same.webp", "Wine A", "Estate A", blockers=["shared_photo_across_slugs"]),
            record(
                "b",
                "same.webp",
                "Wine B",
                "Estate B",
                status="ambiguous",
                blockers=["media_ambiguous", "shared_photo_across_slugs"],
            ),
            record("c", "c.webp", "Wine C", "Estate C"),
        ]
        report = audit_catalog(rows)
        self.assertEqual(report["catalog"]["records_blocked"], 2)
        self.assertEqual(
            report["catalog"]["blocker_counts"],
            {"media_ambiguous": 1, "shared_photo_across_slugs": 2},
        )
        self.assertEqual(report["catalog"]["index_metadata_inconsistencies"], [])
        conflict = report["shared_photo_conflicts"]["groups"][0]
        self.assertEqual(conflict["photo_name"], "same.webp")
        self.assertEqual(conflict["differing_identity_fields"], ["name", "winery"])
        self.assertEqual([item["slug"] for item in conflict["records"]], ["a", "b"])

    def test_flags_stale_index_metadata(self):
        row = record("a", "a.webp", "Wine", "Estate", status="unresolved")
        report = audit_catalog([row])
        inconsistency = report["catalog"]["index_metadata_inconsistencies"][0]
        self.assertEqual(inconsistency["expected_blockers"], ["media_unresolved"])
        self.assertFalse(inconsistency["expected_index_ready"])

    def test_alias_audit_tolerates_transliteration_but_exposes_missing_identity(self):
        row = record(
            "pino-nuar-2025",
            "wine.webp",
            "Пино Нуар, 2025",
            "Винодельня Братьев Мельниковых",
        )
        self.assertEqual(unmatched_query_tokens("Pinot Noir 2025", row), [])
        self.assertEqual(unmatched_query_tokens("Tabia Pinot Noir 2025", row), ["tabia"])
        report = audit_catalog([row], {"pino-nuar-2025": ["Табия Пино Нуар 2025"]})
        gap = report["query_aliases"]["aliases_with_identity_gaps"][0]
        self.assertEqual(gap["missing_tokens"], ["tabiya"])
        self.assertEqual(gap["catalog_identity"]["winery"], "Винодельня Братьев Мельниковых")

    def test_reports_alias_target_gaps_and_is_input_order_independent(self):
        a = record("a", "a.webp", "Alpha", "Estate")
        b = record("b", "b.webp", "Beta", "Estate")
        aliases = {"missing": ["Missing Wine"], "a": ["Alpha"]}
        first = audit_catalog([b, a], aliases)
        second = audit_catalog([a, b], dict(reversed(list(aliases.items()))))
        self.assertEqual(first, second)
        self.assertEqual(
            first["query_aliases"]["missing_target_slugs"],
            [{"slug": "missing", "aliases": ["Missing Wine"]}],
        )


class CatalogAuditLoadingTests(unittest.TestCase):
    def test_strict_loaders_reject_wrong_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            catalog.write_text(json.dumps(["not", "an", "object"]) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_jsonl(catalog)

            aliases = root / "aliases.json"
            aliases.write_text(json.dumps({"slug": "not-a-list"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_label_aliases(aliases)


if __name__ == "__main__":
    unittest.main()
