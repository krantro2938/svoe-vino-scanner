# CIFR inference service

Offline, CPU-first FastAPI service for the hackathon evaluator and mobile client. It
uses a conservative RootSIFT geometric match first, a trained DINOv2 retrieval fallback
for uncertain single-bottle views, narrow centre visual matching for multi-bottle scenes,
and local Russian/English OCR. No model is downloaded at startup.

OCR runs three complementary regions first and expands to five more only when
the lexical evidence is below `CIFR_OCR_MIN_SCORE`. Result confidence is bounded
and capped when producer/vintage evidence conflicts, a match depends on reviewed
aliases, the winner is not visually indexed, or the top candidates are nearly
tied. It is evidence strength, not an aggregate F1 score.

## Run

```bash
cd services/inference
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Python 3.13 enables the pinned ONNX Runtime wheel; Docker already uses 3.13. On
Python 3.14 the optional neural fallback is skipped and the remaining stages still
work. Install the `tesseract` executable through the OS package manager. The pinned
trained-data files are included under `data/tessdata`; set `CIFR_OCR_ENABLED=0`
to exercise the visual-only fallback.

From a repository checkout, the expanded catalogue and indexes are discovered
automatically when present. In another deployment, set `CIFR_CATALOG_PATH`,
`CIFR_INDEX_MANIFEST`, `CIFR_LOCAL_FEATURE_INDEX`, and `CIFR_NEURAL_INDEX_DIR`.

```bash
curl -F 'image=@../../Датасет/eval/queries/019c68d0.jpg' \
  http://127.0.0.1:8080/v1/eval/predict
```

## Build the baseline index

The image directory must contain the filenames in the catalog's `Название фото`/`photo_name` field. If the source archive renames files, use the data pipeline's reconciled image directory.

```bash
PYTHONPATH=. .venv/bin/python tools/build_index.py \
  --catalog ../../Датасет/strapi_output0709.csv \
  --images-dir ../../data/catalog-images \
  --output data/index_manifest.json
```

The repository paths are discovered automatically. The manifest has `descriptor_version`, `model_version`, and `items`; each item contains `slug` and one or more normalized `descriptors`. The loader rejects incompatible or mixed-dimension descriptors rather than silently returning bad results.

Build the high-confidence geometric gallery after creating the expanded catalogue:

```bash
PYTHONPATH=. .venv/bin/python tools/build_local_index.py \
  --catalog ../../data/generated/expanded/catalog.jsonl \
  --references ../../data/generated/expanded/training_references.jsonl \
  --images-dir ../../data/generated/expanded/images \
  --output ../../data/generated/expanded/local_features.npz
```

The DINOv2 experiment and ONNX export commands are documented in
[`training/README.md`](../../training/README.md). Set `CIFR_LOCAL_FEATURES_ENABLED=0`
or `CIFR_NEURAL_ENABLED=0` for ablations. Every optional index validates its checksum,
catalogue identity, schema and model revision before becoming active.

## API

- `POST /v1/eval/predict`: multipart field `image`; exact flat `{"slug":"..."}` response.
- `POST /v1/search`: rich status, wine card, calibrated-like baseline confidence, margin, timing, version, and optional photo-quality hint.
- `GET /v1/wines/{slug}`: local catalog card.
- `POST /v1/sommelier/pairing`: deterministic pairing from `slug`, `dish`, optional `sauce`, and optional `preference`.
- `GET /health/live` and `GET /health/ready`: liveness and catalog/index readiness.

Images are decoded from their bytes rather than filename or multipart MIME type. Upload bytes, decoded pixels, and minimum dimensions are bounded. Errors are small structured JSON documents.

## Tests

```bash
.venv/bin/pytest
```

The visual descriptor is the packaging-safe fallback. OCR ranking is strictly
catalog-grounded, exposes its top-five candidate evidence internally, and label-name
changes are recorded in the reviewed `data/catalog/label_aliases.json` file rather
than hidden in request code. The evaluator endpoint remains the exact flat
`{"slug":"..."}` contract.
