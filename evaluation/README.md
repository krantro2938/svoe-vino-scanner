# Evaluation

## Field-like benchmark (current headline metrics)

The public set has three unlabelled photos, two of them of wines that are not in
the catalogue, so accuracy is measured on rendered field-like photos instead:

```bash
P=services/inference/.venv/bin/python
# 1. Render: label close-up, cylinder curvature, perspective, neighbour bottles,
#    glare, white balance, blur, noise, JPEG. v2 trains the ranker, v1 evaluates.
$P evaluation/field_benchmark.py --count 1500 --seed field-v2 --output evaluation/artifacts/field-v2
$P evaluation/field_benchmark.py --count 700  --seed field-v1 --output evaluation/artifacts/field-v1
# 2. Cache OCR words and per-candidate features (embeddings/RootSIFT are cached).
for v in v1 v2; do
  $P evaluation/cache_ocr.py --queries evaluation/artifacts/field-$v/queries.jsonl \
     --output evaluation/artifacts/field-$v/ocr_words.jsonl
  $P evaluation/dump_fusion_features.py --queries evaluation/artifacts/field-$v/queries.jsonl \
     --ocr-words evaluation/artifacts/field-$v/ocr_words.jsonl \
     --output evaluation/artifacts/field-$v/fusion_features.jsonl
done
# 3. Studio->field SigLIP adapter (selected on held-out v2 wines) and the ranker.
training/.venv/bin/python training/fit_adapter.py --gallery data/generated/expanded/siglip/gallery.npz \
  --train-cache evaluation/artifacts/field-v2/fusion_features.cache.jsonl --train-queries evaluation/artifacts/field-v2/queries.jsonl \
  --eval-cache evaluation/artifacts/field-v1/fusion_features.cache.jsonl --eval-queries evaluation/artifacts/field-v1/queries.jsonl \
  --l2 20 --epochs 30 --output data/generated/expanded/siglip/adapter.npy
$P training/fit_fusion.py --features evaluation/artifacts/field-v2/fusion_features.jsonl \
  --eval evaluation/artifacts/field-v1/fusion_features.jsonl --mismatch 0.2 \
  --output services/inference/data/fusion.json
# 4. End-to-end through the service engine (live OCR, real latency) + F1 report.
$P evaluation/run_engine_benchmark.py --queries evaluation/artifacts/field-v1/queries.jsonl \
  --output evaluation/artifacts/field-v1-final.json
python3 evaluation/evaluate.py --queries evaluation/artifacts/field-v1/queries.jsonl \
  --predictions evaluation/artifacts/field-v1-final.predictions.jsonl \
  --candidates evaluation/artifacts/field-v1-final.candidates.jsonl \
  --catalog data/generated/expanded/catalog.jsonl \
  --report-json evaluation/artifacts/field-v1-report.json --report-md evaluation/artifacts/field-v1-report.md
```

`--mismatch 0.2` swaps the true wine's visual/geometric features with a close
sibling's in 20% of training queries. Real shelf labels often differ from the small
catalogue photo (new vintage, redesign), and without this the ranker becomes
confidently wrong in exactly that case. Both labelled real bottles available (the
public Massandra photo and a second Massandra bottle visible in it) favoured it.

The earlier studio-framed synthetic set (below) remains for regression history;
its numbers are much easier than real photos and are not the headline.

## Earlier synthetic set

This directory is independent of the organizer-provided `Датасет/eval/participant_test.sh`; that file is never modified.

## 1. Build a labeled synthetic set

Prepare a CSV, TSV, JSON, or JSONL reference manifest with at least `image_path` and `slug`. Optional columns are:

- `leakage_group` or `source_group`: keep related labels/producers in one split;
- `hard_negative_group`, `label_family`, or `confusion_group`: report confusable-family metrics;
- `split`: explicitly select `train`, `validation`, or `test` for a connected group.

Then generate deterministic field-like crops, rotations, lighting/contrast shifts, blur, edge occlusion, and JPEG degradation:

```bash
services/inference/.venv/bin/python evaluation/synthetic_dataset.py \
  --references evaluation/references.tsv \
  --images-dir data/catalog-images \
  --output-dir evaluation/artifacts/synthetic-v1 \
  --variants 3 \
  --augmentation-seed photos-1 \
  --split-seed frozen-v1 \
  --splits validation=0.5,test=0.5
```

The output contains `queries.jsonl`, transformed JPEGs under `images/<split>/`, and `dataset.json` with the exact seeds and counts. The output directory must be new or empty, preventing an accidental mix of runs.

Split isolation is stricter than row-level random splitting. Every connected component sharing a ground-truth slug, identical source bytes, or an explicit leakage group is assigned as one unit. `augmentation-seed` affects images and query IDs but not split assignment, so a new augmentation run cannot move the same source or class across splits. Conflicting explicit splits and identical source bytes with different labels are rejected.

Pillow is the only image dependency and is already pinned in the inference environment. Reference images are never modified.

## 2. Exercise the running service

The benchmark sends requests sequentially using the organizer's multipart field and response contract. It does not overwrite a prior result unless `--force` is explicit.

```bash
python3 evaluation/benchmark_service.py \
  --images-dir evaluation/artifacts/synthetic-v1 \
  --manifest evaluation/artifacts/synthetic-v1/queries.jsonl \
  --endpoint http://127.0.0.1:8080/v1/eval/predict \
  --output evaluation/artifacts/predictions.jsonl
```

Use `--limit 1` for a quick smoke request. The emitted JSONL retains the five official submission fields. If the service returns a ranked list, `candidate_slugs` is included for Recall@5.

For the final organizer-compatible run, use their script exactly as documented in `Датасет/eval/README.md`.

## 3. Validate and score predictions

```bash
python3 evaluation/evaluate.py \
  --queries evaluation/artifacts/synthetic-v1/queries.jsonl \
  --predictions evaluation/artifacts/predictions.jsonl \
  --catalog 'Датасет/strapi_output0709.csv' \
  --images-dir evaluation/artifacts/synthetic-v1 \
  --report-json evaluation/artifacts/report.json \
  --report-md evaluation/artifacts/report.md
```

Exit codes are `0` for a valid submission, `1` for validation errors (or warnings with `--fail-on-warnings`), and `2` for unreadable/malformed inputs.

### Query labels

The public `queries.tsv` has no ground truth. In that case the report says `unlabeled_validation` and deliberately omits Accuracy/F1. It still checks:

- prediction schema, query coverage, uniqueness, and order;
- image-path and SHA-256 consistency;
- predicted and candidate slugs against the catalog;
- null-result rate and latency p50/p95/max.

For an internal labeled set, add any one of these columns to the query manifest: `expected_slug`, `true_slug`, `ground_truth_slug`, `label`, or `slug`. The scorer then reports exact Accuracy@1, micro-F1, macro-F1, and per-class results. In this single-label multiclass task, micro-F1 equals exact accuracy.

When the manifest contains `split`, metrics are also sliced by split. When it contains `hard_negative_group`, `label_family`, `confusion_group`, `group_id`, or `group`, the scorer reports per-group accuracy and Recall@5. A group containing at least two distinct ground-truth slugs is additionally summarized as a hard-negative slice. An explicit `is_hard_negative`/`hard_negative` boolean is supported when group labels are unavailable.

Top-5 candidates may be embedded in prediction rows under `candidate_slugs`, `top_5`, `top5`, `top_k`, `candidates`, or `matches`. A candidate may be a slug string or an object with a `slug`. Alternatively pass a separate JSONL/JSON/CSV/TSV file with `--candidates`; table values may be JSON arrays or pipe/comma-separated slugs. Recall@5 is reported only for labeled queries that have candidate data, and candidate coverage is always shown so partial data cannot be mistaken for full evaluation.

## Tests

```bash
services/inference/.venv/bin/python -m unittest discover -s tests/evaluation -v
```

## 4. Prepare training references and centred-bottle scenes

Build a reference manifest from a generated catalogue and its local images. This checks
image decoding and dimensions, excludes rows explicitly marked `index_ready=false`, and
quarantines **all** labels whose image bytes also belong to another label. Rejected rows
and reasons are retained in the companion `.report.json` file. It never guesses a label.

```bash
services/inference/.venv/bin/python evaluation/prepare_training_references.py \
  --catalog data/generated/catalog/catalog.jsonl \
  --images-dir data/catalog-images \
  --output evaluation/artifacts/official-references.jsonl

services/inference/.venv/bin/python evaluation/synthetic_dataset.py \
  --references evaluation/artifacts/official-references.jsonl \
  --images-dir data/catalog-images \
  --output-dir evaluation/artifacts/training-v3 \
  --variants 4 --shelf-variants 2 \
  --augmentation-seed field-v3 --split-seed frozen-v2 \
  --splits train=0.8,validation=0.1,test=0.1
```

For the final merged catalogue, preserve alternate verified photographs by using the
merger's reference manifest directly (rather than rebuilding it from one catalogue
image per wine). The final expanded build uses these exact paths and seeds:

```bash
services/inference/.venv/bin/python evaluation/synthetic_dataset.py \
  --references data/generated/expanded/training_references.jsonl \
  --images-dir data/generated/expanded/images \
  --output-dir evaluation/artifacts/training-expanded-v3 \
  --variants 4 --shelf-variants 2 \
  --augmentation-seed wine-photos-v3 --split-seed wine-producer-v1 \
  --splits train=0.8,validation=0.1,test=0.1

services/inference/.venv/bin/python evaluation/audit_training_dataset.py \
  --dataset evaluation/artifacts/training-expanded-v3 \
  --output evaluation/artifacts/training-expanded-v3/audit.json
```

A declared `source_sha256` must match the actual reference bytes; stale downloads fail
before generation. Duplicate reference rows cannot conceal conflicting split labels.

This produces four individually degraded references and two three-bottle scenes per
source. In every three-bottle scene the correct bottle occupies the horizontal centre;
two other distinct labels are placed on either side. Distractors from the same producer
are preferred when available. Distractors always come from the **same split**, so test
images cannot leak into training as background bottles. A split with fewer than three
classes cannot produce shelf scenes and fails before writing files.

Generator version 3 adds mild projective distortion and soft diagonal glare to single-reference views, retaining exact seeded transform parameters. Each generated image records a SHA-256 checksum. Images are at most 640 pixels per side. Shelf scenes preserve source transparency,
include lighting, blur and JPEG variation, and record each visible label, its source
hash/provenance and rendered-reference bounding box in `objects`. Those boxes describe
reference extents, not human-annotated bottle silhouettes. `hard_negative_slugs` lists
the two explicit distractors; `expected_slug` always describes the middle bottle.
Catalogue producer names populate both `leakage_group` and `hard_negative_group`, keeping
related labels together across train/validation/test. Hash assignment is deterministic;
actual split proportions can differ from requested weights because producer groups are
indivisible. Changing augmentations never changes split membership.

For embedding training, use train rows' `image_path` as anchors, their
`source_image_path` as positive reference images, and non-target `objects` references
as hard negatives. Query image paths resolve relative to the generated dataset;
source image paths resolve relative to `--images-dir`. Train only on `split=train` and
select hyperparameters using validation. Preserve the test split until the final run.
This class-disjoint split tests retrieval generalisation, so a catalogue gallery can
contain held-out classes at evaluation but must not contribute gradient updates.
A fixed-class classification model instead needs independent real photographs per
class and a separately designed split.

These are **synthetic catalogue derivatives**, not independent customer photos and not
a measured accuracy improvement. They supplement, rather than replace, a held-out set
of real phone photos with centred targets, glare, shelf clutter and similar labels.
The generator creates training data; it does not fit or evaluate model weights.

Audit the completed dataset before training:

```bash
services/inference/.venv/bin/python evaluation/audit_training_dataset.py \
  --dataset evaluation/artifacts/training-v3 \
  --output evaluation/artifacts/training-v3/audit.json
```

The audit checks query file checksums, duplicate IDs, class/source/producer leakage
(including distractor images), three distinct shelf labels, image bounds, and whether
the target bounding box is nearest to the image centre.
