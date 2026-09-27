# Wine training data and centre-target recognition

The competition gives recognition quality 50% of the score and stresses near-identical labels, real phone photographs, and image normalization. A larger catalogue alone does not establish accuracy. This pipeline keeps the organiser identities intact, adds public website records separately, and rejects contradictory reference labels before generating training examples.

## Reproduce the expanded dataset

Run from the repository root. The scraper resumes its existing snapshot; use another output directory when intentionally capturing a new website version.

```bash
services/inference/.venv/bin/python scripts/scrape_wine_catalog.py
python3 scripts/merge_wine_catalog.py \
  --snapshot data/external/vino-svoe/catalog.jsonl \
  --snapshot-root data/external/vino-svoe \
  --output data/generated/expanded
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

Merge and generation output directories must be new or empty. They are deliberately excluded from Git along with downloaded images. Scripts, tests, and the compact delivery report are tracked; retain the generated local data when preparing a training machine.

The expanded catalogue contains every organiser slug. Website metadata for matching slugs is stored under `website_snapshot` without overwriting the original identity. New exact website slugs are added. Exact image hashes and canonical website upload paths identify shared-image conflicts, including resized copies with different file hashes. Conflicting identities remain in the catalogue but are excluded from the training references and visual index. An ambiguous or missing archive reference is recovered only when the exact website slug, normalized wine name and winery agree, the downloaded image is verified, and its hash and canonical upload identity do not belong to another wine. Recovery provenance is recorded separately; shared-image conflicts remain quarantined.

Verified alternate website views of the same slug may be included as additional reference images. Every reference carries its source and checksum. Matching a website image to an organiser image uses the exact slug and canonical upload identity, rather than wine-name similarity.

## Training format and split policy

`training_references.jsonl` supplies clean source photographs. `queries.jsonl` supplies generated examples. Each example identifies its `expected_slug`, `split`, source hash, transformation parameters, and whether it is synthetic. Three-bottle examples also identify all bottle slugs, the centre target, and their reference-extent bounding boxes. These boxes are not hand-annotated bottle segmentation masks.

Use only `split=train` for gradient updates and hard-negative mining. The same wine, source photograph, and producer family stay within one split. Distractor bottles come from the target split, so a held-out label cannot leak into training through the background. The proportions are approximate because producers are indivisible groups. Keep `test` frozen; use `validation` for model selection.

The intended model task is metric-learning retrieval: a query embedding should approach its matching reference and move away from other wines, especially wines by the same producer. A classifier trained only on train slugs cannot classify the deliberately disjoint held-out classes. For retrieval evaluation, encode the held-out catalogue references as the search gallery without using them for gradient updates; retrieve each held-out query against the full gallery.

Four single-reference derivatives and two centre-target shelf composites per clean source provide lighting, blur, compression, viewpoint and distractor stress cases. The generated shelf backgrounds are simple and the bounding boxes describe source extents; they are useful augmentations, not substitutes for genuine shelf photographs. Source and query checksums, seeds and audits make a generated run traceable.

## What still determines competition performance

Collect an independently labelled phone-photo set covering hard pairs: vintage, grape, sweetness and edition changes, plus shelves, glare, angles, and partially obscured bottles. Keep every physical photo session together. Do not report accuracy on synthetic copies of reference images as real-world accuracy. The three supplied public photographs do not include verified ground truth.

The pinned DINOv2 projection experiment, exact environment and ONNX export are documented
in [`training/README.md`](../training/README.md). RootSIFT plus validation-selected routed
fallback now reaches 96.10% Top-1 on the full synthetic validation split. Collect an
independently labelled real-phone test set before treating that as product or private-set
accuracy; the target remains the brief's 90-100% range, not a guarantee.

## Expanded runtime index

The expanded catalogue is now preferred automatically when its generated artifacts exist.
Build its packaging-safe descriptor index with:

```bash
cd services/inference
PYTHONPATH=. .venv/bin/python tools/build_index.py \
  --catalog ../../data/generated/expanded/catalog.jsonl \
  --images-dir ../../data/generated/expanded/images \
  --output ../../data/generated/expanded/index_manifest.json
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Evaluate the expanded index before freezing a competition configuration: adding similar wines can change Top-1 even when it improves catalogue coverage.
