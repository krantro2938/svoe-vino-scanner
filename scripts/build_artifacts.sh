#!/usr/bin/env bash
# Rebuild every runtime artifact from the organiser package in Датасет/.
#
#   scripts/build_artifacts.sh            # SNAPSHOT defaults to the 2026-09-27 site snapshot
#
# Steps: organiser catalogue -> media extraction -> merge with the public site
# snapshot (scraped with scripts/scrape_wine_catalog.py) -> SigLIP 2
# ONNX export -> SigLIP gallery -> RootSIFT index. The fitted fusion weights are
# (fusion.json) and the SigLIP adapter (adapter.npy) are versioned in
# services/inference/data/ and need no retraining.
set -euo pipefail
cd "$(dirname "$0")/.."

SERVICE_PY=${SERVICE_PY:-services/inference/.venv/bin/python}
TRAINING_PY=${TRAINING_PY:-training/.venv/bin/python}
EXPANDED=data/generated/expanded

step() { printf '\n== %s\n' "$*"; }

step "Organiser catalogue and media"
python3 scripts/build_catalog.py
python3 scripts/extract_catalog_media.py

SNAPSHOT=${SNAPSHOT:-data/external/vino-svoe-20260927}
if [ ! -f "$SNAPSHOT/catalog.jsonl" ]; then
  echo "Missing $SNAPSHOT; create it with:" >&2
  echo "  $SERVICE_PY scripts/scrape_wine_catalog.py --output $SNAPSHOT" >&2
  exit 1
fi
step "Merge organiser catalogue with public snapshot ${SNAPSHOT}"
"$SERVICE_PY" scripts/merge_wine_catalog.py --snapshot "$SNAPSHOT/catalog.jsonl" \
  --snapshot-root "$SNAPSHOT" --output "$EXPANDED"

step "Export SigLIP 2 image tower to ONNX"
"$TRAINING_PY" training/export_siglip.py --output "$EXPANDED/siglip"

# Versioned studio->field adapter (training/fit_adapter.py), applied at query time.
cp services/inference/data/adapter.npy services/inference/data/adapter.json "$EXPANDED/siglip/"

step "Embed catalogue references (SigLIP gallery)"
(cd services/inference && "../../$SERVICE_PY" tools/build_siglip_index.py \
  --catalog "../../$EXPANDED/catalog.jsonl" --images-dir "../../$EXPANDED/images" \
  --encoder-dir "../../$EXPANDED/siglip")

step "RootSIFT geometric verification index"
(cd services/inference && PYTHONPATH=. "../../$SERVICE_PY" tools/build_local_index.py \
  --catalog "../../$EXPANDED/catalog.jsonl" --references "../../$EXPANDED/training_references.jsonl" \
  --images-dir "../../$EXPANDED/images" --output "../../$EXPANDED/local_features.npz")

step "Done"
ls -la "$EXPANDED/siglip" "$EXPANDED/local_features.npz"
