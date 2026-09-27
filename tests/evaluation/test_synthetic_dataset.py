import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation"))

from evaluate import InputError  # noqa: E402
from synthetic_dataset import (  # noqa: E402
    _render,
    _variant_parameters,
    assign_splits,
    generate_dataset,
    load_sources,
    parse_split_weights,
)


def write_ppm(path: Path, color: tuple[int, int, int]) -> None:
    width, height = 24, 20
    path.write_bytes(
        f"P6\n{width} {height}\n255\n".encode("ascii")
        + bytes(color) * width * height
    )


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class SyntheticDatasetTests(unittest.TestCase):
    def test_transforms_are_reproducible_and_seed_does_not_change_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "references"
            images.mkdir()
            write_ppm(images / "a1.ppm", (220, 10, 10))
            write_ppm(images / "a2.ppm", (180, 20, 20))
            write_ppm(images / "b.ppm", (10, 220, 10))
            (root / "references.tsv").write_text(
                "image_path\tslug\tleakage_group\thard_negative_group\n"
                "a1.ppm\twine-a\tfamily-a\tsame-label\n"
                "a2.ppm\twine-a\tfamily-a\tsame-label\n"
                "b.ppm\twine-b\tfamily-b\tsame-label\n",
                encoding="utf-8",
            )
            weights = parse_split_weights("validation=0.5,test=0.5")
            first = root / "first"
            repeat = root / "repeat"
            other_seed = root / "other-seed"
            generate_dataset(
                root / "references.tsv", images, first,
                variants_per_image=2, augmentation_seed="photos-1", split_seed="frozen", split_weights=weights,
            )
            generate_dataset(
                root / "references.tsv", images, repeat,
                variants_per_image=2, augmentation_seed="photos-1", split_seed="frozen", split_weights=weights,
            )
            generate_dataset(
                root / "references.tsv", images, other_seed,
                variants_per_image=2, augmentation_seed="photos-2", split_seed="frozen", split_weights=weights,
            )

            first_rows = read_jsonl(first / "queries.jsonl")
            repeat_rows = read_jsonl(repeat / "queries.jsonl")
            other_rows = read_jsonl(other_seed / "queries.jsonl")
            self.assertEqual(first_rows, repeat_rows)
            for row in first_rows:
                relative = row["image_path"]
                self.assertEqual((first / relative).read_bytes(), (repeat / relative).read_bytes())
                self.assertEqual(row["sha256"], hashlib.sha256((first / relative).read_bytes()).hexdigest())
                self.assertIn("perspective_x", row["transform"])
                self.assertGreater(row["transform"]["glare_alpha"], 0)
                self.assertEqual(row["expected_slug"], "wine-a" if "a" in row["source_image_path"] else "wine-b")
            first_splits = {(row["source_sha256"], row["split"]) for row in first_rows}
            other_splits = {(row["source_sha256"], row["split"]) for row in other_rows}
            self.assertEqual(first_splits, other_splits)
            self.assertNotEqual(
                {row["query_id"] for row in first_rows},
                {row["query_id"] for row in other_rows},
            )
            wine_a_splits = {row["split"] for row in first_rows if row["expected_slug"] == "wine-a"}
            self.assertEqual(len(wine_a_splits), 1)

    def test_explicit_group_prevents_cross_split_and_conflicting_declarations_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_ppm(root / "a.ppm", (1, 2, 3))
            write_ppm(root / "b.ppm", (4, 5, 6))
            manifest = root / "references.tsv"
            manifest.write_text(
                "image_path\tslug\tleakage_group\tsplit\n"
                "a.ppm\ta\tshared\tvalidation\n"
                "b.ppm\tb\tshared\ttest\n",
                encoding="utf-8",
            )
            sources = load_sources(manifest, root)
            with self.assertRaisesRegex(InputError, "conflicting declared splits"):
                assign_splits(sources, {"validation": 0.5, "test": 0.5}, split_seed="fixed")

    def test_identical_image_bytes_cannot_have_different_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_ppm(root / "same.ppm", (1, 2, 3))
            (root / "copy.ppm").write_bytes((root / "same.ppm").read_bytes())
            manifest = root / "references.tsv"
            manifest.write_text(
                "image_path\tslug\nsame.ppm\ta\ncopy.ppm\tb\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(InputError, "conflicting slugs"):
                load_sources(manifest, root)

    def test_shelf_target_is_centered_and_all_distractors_stay_in_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "references.jsonl"
            references = []
            for i in range(6):
                name = f"{i}.ppm"
                write_ppm(root / name, (i * 30, 50, 10))
                references.append({"image_path": name, "slug": f"wine-{i}",
                                   "split": "train" if i < 3 else "test",
                                   "source_url": f"https://example.org/wine-{i}",
                                   "hard_negative_group": "family"})
            manifest.write_text("\n".join(json.dumps(row) for row in references))
            first = root / "generated"
            repeat = root / "repeat"
            for output in (first, repeat):
                summary = generate_dataset(manifest, root, output, variants_per_image=1,
                                           shelf_variants_per_image=1,
                                           split_weights={"train": 0.5, "test": 0.5})
            self.assertEqual(summary["query_count"], 12)
            rows = read_jsonl(first / "queries.jsonl")
            for row in rows:
                self.assertEqual((first / row["image_path"]).read_bytes(),
                                 (repeat / row["image_path"]).read_bytes())
                if row["scene_type"] == "single_reference":
                    continue
                self.assertEqual(len(row["objects"]), 3)
                target = next(item for item in row["objects"] if item["is_target"])
                self.assertEqual(target["slug"], row["expected_slug"])
                self.assertEqual(target["provenance"]["source_url"], row["provenance"]["source_url"])
                x1, _, x2, _ = target["bbox_xyxy"]
                self.assertLessEqual(abs((x1 + x2) / 2 - row["width"] / 2), 1)
                allowed = {item["slug"] for item in references if item["split"] == row["split"]}
                self.assertTrue({item["slug"] for item in row["objects"]} <= allowed)
                self.assertEqual(len(set(row["hard_negative_slugs"])), 2)
                self.assertNotIn(row["expected_slug"], row["hard_negative_slugs"])

    def test_shelf_small_split_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_ppm(root / "a.ppm", (1, 2, 3))
            manifest = root / "references.tsv"
            manifest.write_text("image_path\tslug\na.ppm\ta\n")
            output = root / "generated"
            with self.assertRaisesRegex(InputError, "at least three"):
                generate_dataset(manifest, root, output, shelf_variants_per_image=1)
            self.assertFalse(output.exists())

    def test_perspective_and_glare_change_rendered_pixels(self):
        from PIL import Image, ImageDraw
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (180, 300), (10, 20, 30))
            ImageDraw.Draw(image).rectangle((50, 110, 135, 210), fill=(190, 120, 40))
            image.save(root / "bottle.png")
            manifest = root / "references.tsv"
            manifest.write_text("image_path\tslug\nbottle.png\tbottle\n")
            source = load_sources(manifest, root)[0]
            base = _variant_parameters(source, "fixed", 0)
            base.update(perspective_x=0, perspective_y=0, glare_alpha=0)
            _render(source, root / "base.jpg", base)
            _render(source, root / "perspective.jpg", {**base, "perspective_x": 0.065})
            _render(source, root / "glare.jpg", {**base, "glare_alpha": 46})
            self.assertNotEqual((root / "base.jpg").read_bytes(), (root / "perspective.jpg").read_bytes())
            self.assertNotEqual((root / "base.jpg").read_bytes(), (root / "glare.jpg").read_bytes())

    def test_stale_source_hash_and_duplicate_split_conflict_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_ppm(root / "a.ppm", (1, 2, 3))
            manifest = root / "references.jsonl"
            manifest.write_text(json.dumps({"image_path": "a.ppm", "slug": "a", "source_sha256": "0" * 64}))
            with self.assertRaisesRegex(InputError, "source_sha256 mismatch"):
                load_sources(manifest, root)
            manifest.write_text("\n".join(json.dumps({"image_path": "a.ppm", "slug": "a", "split": split})
                                         for split in ("train", "test")))
            with self.assertRaisesRegex(InputError, "conflicting declared splits"):
                load_sources(manifest, root)

    def test_split_weights_are_strict(self):
        self.assertEqual(parse_split_weights("validation=0.2,test=0.8"), {"validation": 0.2, "test": 0.8})
        with self.assertRaisesRegex(InputError, "sum to 1.0"):
            parse_split_weights("validation=0.2,test=0.7")


if __name__ == "__main__":
    unittest.main()
