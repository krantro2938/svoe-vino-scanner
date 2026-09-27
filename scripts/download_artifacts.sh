#!/usr/bin/env bash
# Download the prebuilt runtime artifacts (SigLIP ONNX + gallery, RootSIFT index,
# expanded catalogue and reference photos) into data/generated/expanded/.
# Rebuild them from the organiser package instead with scripts/build_artifacts.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO=${REPO:-krantro2938/svoe-vino-scanner}
TAG=${TAG:-artifacts-v1}
FILE=cifr-artifacts.tar.zst
BASE="https://github.com/$REPO/releases/download/$TAG"

curl -fL --retry 3 -o "/tmp/$FILE" "$BASE/$FILE"
curl -fL --retry 3 -o "/tmp/$FILE.sha256" "$BASE/$FILE.sha256"
(cd /tmp && sha256sum -c "$FILE.sha256")
tar --zstd -xf "/tmp/$FILE" -C .
rm -f "/tmp/$FILE" "/tmp/$FILE.sha256"
echo "Artifacts ready in data/generated/expanded/"
