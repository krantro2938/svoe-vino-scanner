import tempfile
import unittest
import json
import binascii
from pathlib import Path

from scripts.build_catalog import (
    ArchiveEntry,
    archive_canonical_key,
    build_media_indexes,
    load_catalog,
    normalized_media_key,
    parse_7z_slt,
    reconcile_media,
)
from scripts.extract_catalog_media import (
    ExtractionItem,
    _install_verified,
    load_extraction_plan,
    validate_archive_path,
    validate_photo_name,
)


class NormalizationTests(unittest.TestCase):
    def test_cyrillic_name_matches_strapi_spelling(self):
        self.assertEqual(normalized_media_key("Спуманте белый брют.webp"), "spumante_belyj_bryut")

    def test_numero_sign_is_dropped_like_strapi(self):
        self.assertEqual(normalized_media_key("Бленд №4.webp"), "blend_4")

    def test_removes_rendition_and_hash(self):
        self.assertEqual(
            archive_canonical_key("thumbnail_Spumante_belyj_bryut_d34e854a7b.webp"),
            "spumante_belyj_bryut",
        )


class ArchiveTests(unittest.TestCase):
    def test_parse_7z_slt_ignores_archive_header_and_directories(self):
        output = """
Path = bundle.part1.rar
Type = Rar5

Path = root/uploads
Folder = +
Size = 0

Path = root/uploads/Wine_Name_0123456789.webp
Folder = -
Size = 1234
Packed Size = 1200
Modified = 2026-01-02 03:04:05.0000000
CRC = ABCDEF12
"""
        self.assertEqual(
            parse_7z_slt(output),
            [
                ArchiveEntry(
                    path="root/uploads/Wine_Name_0123456789.webp",
                    size=1234,
                    packed_size=1200,
                    modified="2026-01-02 03:04:05.0000000",
                    crc="ABCDEF12",
                )
            ],
        )

    def test_reconciliation_prefers_original_to_thumbnail(self):
        entries = [
            ArchiveEntry("root/uploads/thumbnail_DSC_09173_4a9ff95cc2.webp", 100),
            ArchiveEntry("root/uploads/DSC_09173_4a9ff95cc2.webp", 1_000),
        ]
        canonical, compact, by_path = build_media_indexes(entries)
        match = reconcile_media("DSC09173.webp", canonical, compact, by_path, {})
        self.assertEqual(match.status, "resolved")
        self.assertEqual(match.method, "separator_insensitive_name")
        self.assertEqual(match.selected.path, entries[1].path)

    def test_multiple_original_uploads_are_ambiguous(self):
        entries = [
            ArchiveEntry("root/uploads/wine_name_1111111111.webp", 1_000),
            ArchiveEntry("root/uploads/wine_name_2222222222.webp", 2_000),
        ]
        canonical, compact, by_path = build_media_indexes(entries)
        match = reconcile_media("wine-name.webp", canonical, compact, by_path, {})
        self.assertEqual(match.status, "ambiguous")
        self.assertEqual(match.selected.path, entries[1].path)

    def test_embedded_source_extension_can_match_transcoded_upload(self):
        entry = ArchiveEntry("root/uploads/random_name_png_1111111111.webp", 1_000)
        canonical, compact, by_path = build_media_indexes([entry])
        match = reconcile_media("random-name.png", canonical, compact, by_path, {})
        self.assertEqual(match.status, "resolved")
        self.assertEqual(match.method, "normalized_name_with_embedded_extension")

    def test_invalid_override_fails_loudly(self):
        with self.assertRaises(ValueError):
            reconcile_media("wine.webp", {}, {}, {}, {"wine.webp": "missing.webp"})


class CsvTests(unittest.TestCase):
    def test_exact_dedup_cleaning_and_conflict_audit(self):
        header = ",".join(
            [
                "Название вина",
                "Категория",
                "Цвет",
                "Регион",
                "Сорт винограда",
                "Описание",
                "Винодельня",
                "Slug",
                "Название фото",
            ]
        )
        rows = [
            ' Wine,Белое,золотой,Крым,," text ",Estate,wine,wine.webp',
            ' Wine,Белое,золотой,Крым,," text ",Estate,wine,wine.webp',
            'Other,Красное,рубиновый,Крым,Сира,text,Estate,other,wine.webp',
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.csv"
            path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
            records, report = load_catalog(path)
        self.assertEqual(len(records), 2)
        self.assertEqual(report["exact_duplicate_rows_removed"], 1)
        self.assertEqual(records[1]["name"], "Wine")
        self.assertEqual(records[1]["raw_overrides"]["name"], " Wine")
        self.assertEqual(report["missing_fields"]["grapes"]["count"], 1)
        self.assertEqual(report["photo_to_multiple_slugs"][0]["slugs"], ["other", "wine"])


class ExtractionTests(unittest.TestCase):
    def test_rejects_unsafe_destination_and_archive_paths(self):
        for value in ("../wine.webp", "folder/wine.webp", "folder\\wine.webp", "wine\n.webp"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_photo_name(value)
        for value in ("/uploads/wine.webp", "../uploads/wine.webp", "uploads/../wine.webp"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_archive_path(value)

    def test_plan_keeps_only_resolved_rows_and_applies_sorted_limit(self):
        rows = [
            {
                "photo_name": "z.webp",
                "status": "resolved",
                "selected": {"path": "root/uploads/z_1111111111.webp", "size": 1, "crc": "00000000"},
                "slugs": ["z"],
            },
            {"photo_name": "ignored.webp", "status": "ambiguous", "selected": None, "slugs": ["ignored"]},
            {
                "photo_name": "a.webp",
                "status": "resolved",
                "selected": {"path": "root/uploads/a_1111111111.webp", "size": 1, "crc": "00000000"},
                "slugs": ["a"],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            plan = load_extraction_plan(path, limit=1)
        self.assertEqual([item.photo_name for item in plan], ["a.webp"])

    def test_verified_install_uses_exact_catalog_photo_name(self):
        payload = b"catalog-image"
        checksum = f"{binascii.crc32(payload) & 0xFFFFFFFF:08X}"
        item = ExtractionItem(
            photo_name="Точное имя.webp",
            archive_path="root/uploads/source_hash.webp",
            size=len(payload),
            crc32=checksum,
            slugs=("wine",),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.webp"
            source.write_bytes(payload)
            destination = root / item.photo_name
            self.assertEqual(_install_verified(source, destination, item, False), "extracted")
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(_install_verified(source, destination, item, False), "already_verified")


if __name__ == "__main__":
    unittest.main()
