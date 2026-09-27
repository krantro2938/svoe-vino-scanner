# Dataset update — 20 September 2026

## Delivered scope

The supplied case prioritizes recognition quality (50/100), particularly similar labels and photographs taken at shelves. This section describes the reproducible training data, expanded reference coverage, and centre-target annotations used by the trained retrieval update below. It does not claim a guaranteed competition result.

| Item | Count |
| --- | ---: |
| Original organiser wines | 2,103 |
| Public website pages captured | 130 / 130 |
| Website wines captured | 2,069 / 2,069 |
| Website image downloads | 107 |
| Verified original images reused | 1,962 |
| Website image failures | 0 |
| New exact website slugs | 34 |
| Expanded catalogue | 2,137 |
| Ambiguous/missing original references safely recovered | 47 |
| Clean training / visual-index classes | 2,070 |
| Clean reference images, including alternate views | 2,078 |
| Remaining blocked catalogue identities | 67 |

Recovery requires the exact website slug, normalized wine name and winery, a verified image hash, and no conflicting image hash or canonical upload identity. The original identity and recovery provenance are retained. One otherwise recoverable name with differing punctuation remains excluded under this strict policy.

The snapshot is the complete public listing at capture time, not a claim to contain every wine ever produced. Listing metadata includes wine name, winery, category, colour and region; fields absent from listings are not invented. The original catalogue supplies its existing richer metadata.

## Local outputs

- `data/external/vino-svoe/catalog.jsonl`: complete website snapshot with cached HTML, images and provenance.
- `data/generated/expanded/catalog.jsonl`: merged catalogue preserving all organiser slugs.
- `data/generated/expanded/merge_report.json`: added/recovered/blocked identities and source checksums.
- `data/generated/expanded/training_references.jsonl`: clean labelled references.
- `data/generated/expanded/index_manifest.json`: separate CPU visual retrieval index.
- `evaluation/artifacts/training-expanded-v3/`: final expanded synthetic training images, manifest and audit.
- `evaluation/artifacts/training-v3/`: earlier organiser-only dataset, retained for reproducible centre benchmarking. Do not combine it with the expanded run: the split seeds differ.

The final expanded run contains four single-reference variants and two centre-target, three-bottle scenes per source. Transformations include perspective, glare, lighting, blur, crop, rotation, edge occlusion and JPEG degradation. Shelf scenes retain the central target, same-split distractors and source-extent boxes. Seeds and hashes identify the exact run. Producer families, wine identities and source bytes stay within one split, including their use as distractors.

Generated images and the downloaded snapshot are local and excluded from Git. Reproduction commands and the model-training split policy are in [TRAINING_DATA.md](TRAINING_DATA.md).

## Recognition checks and remaining limitation

Centre targeting uses a framing heuristic for wide images and spatial filtering of complete OCR words. It is not a trained bottle detector. The final ensemble identifies itself as `rootsift-dinov2-ocr-v1`.

The three unchanged public photographs retain their previous predicted slugs, with approximately 2.0–2.2 seconds per prediction in the regression check. Their true labels are not supplied, so this is a regression check, not an accuracy measurement. See `data/generated/center-target-benchmark.json`. The separately rebuilt expanded index also returns the same three slugs, with observed prediction times of 1.90, 2.25 and 2.76 seconds; see `data/generated/expanded-public-benchmark.json`.

On 32 deterministic held-out synthetic shelves, the previous visual-only pipeline scored 0/32, centre-focused visual matching scored 3/32, and the full centre-focused OCR+visual pipeline also scored 3/32 (9.375%). No tuning was performed on that subset. Hybrid median prediction time was 296 ms and p95 453 ms, excluding image decoding/model loading, with one OpenMP/OpenBLAS thread. This stress test exposes a weak baseline; it does not establish real-world or private-test accuracy. See `data/generated/center-shelf32-benchmark.json`.

The next quality gate is training an embedding model or bottle detector with the clean train split, selecting changes on validation, and evaluating on independently labelled real phone photographs. Keep the test split frozen. Expanding the catalogue and generating augmentations alone do not satisfy the brief's 90–100% recognition target.

## Trained retrieval update - 21 September 2026

The expanded data now feeds two measured retrieval stages. RootSIFT local features with
FLANN retrieval and RANSAC geometric verification reached 198/200 (99%) on the fixed
validation experiment: 98/100 single-reference and 100/100 centre-target shelf scenes.
The conservative runtime gate accepted 1,132 of all 1,332 validation queries with zero
accepted-match errors. Combined with the old visual fallback it reached 93.02% validation
Top-1 at 97 ms median and 169 ms p95.

A pinned DINOv2-small backbone (`facebook/dinov2-small`, revision
`150c8e7bb7cef2d30ec31b13a517af14840ee3f7`) achieved 57.13% frozen-backbone validation
Top-1. A train-only identity-initialized projection selected at epoch 10 reached 70.72%
and 87.99% Recall@5; no test query was decoded or scored during training or selection.
The selected model is exported to checksum-bound ONNX for the Python 3.13/Docker runtime.

The stages have complementary errors. Validation-selected routing uses DINO for uncertain
single-bottle views and the narrow centre visual path when edge evidence indicates objects
on both sides. Across all validation images, RootSIFT + routed fallback reached
1,280/1,332 = **96.10% Top-1**: 97.18% for single-reference views and 93.92% for
centre-target three-bottle scenes. The three unlabelled public-photo predictions remain
unchanged at 2.54-2.74 seconds in the final untouched organiser-client run through the
full RootSIFT/DINO/OCR service, with OpenMP/BLAS contention bounded to one thread.

Before adding DINO/routing, the once-only frozen synthetic test of RootSIFT + visual
fallback measured 979/1,098 = 89.16%. The test was not rerun after the new fallback was
selected, so there is no uncontaminated final-test number for the promoted ensemble.
All synthetic metrics remain catalogue-derived; an independently photographed and labelled
phone set is still required before claiming private-set or real-world accuracy.

## Automated verification

`PYTHONPATH=services/inference:. services/inference/.venv/bin/python -m pytest -q`

83 tests pass and one training-only test module is skipped under Python 3.14 because
PyTorch is isolated in the Python 3.12 training environment. The ONNX-enabled Python
3.13 inference suite also passes. Coverage includes scraping/parsing, safe image paths,
exact-source recovery, conflicting image quarantine, split isolation, source hashes,
centre annotations, local/neural index validation, OCR word filtering and the API contract.
