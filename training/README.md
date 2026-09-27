# DINOv2 wine embedding experiment

This isolated experiment leaves the inference service unchanged. It uses the official
[Facebook DINOv2-small checkpoint](https://huggingface.co/facebook/dinov2-small/tree/150c8e7bb7cef2d30ec31b13a517af14840ee3f7)
(22M parameters, pinned revision `150c8e7bb7cef2d30ec31b13a517af14840ee3f7`).
The backbone is frozen. A 768-dimensional metric projection is trained using only
training query/reference pairs and training negatives; validation selects the best
checkpoint or retains the untrained baseline. This is projection training, not a
fine-tuned image encoder.

## Reproduce

A separate Python 3.12 environment avoids the service's Python 3.14 dependency limits.
[PyTorch publishes CPU wheels separately](https://docs.pytorch.org/get-started/previous-versions/),
which keep this experiment around 1GB installed without downloading a full CUDA stack.
The machine's RTX4050 has 6GB VRAM but lacks CUDA user-space libraries; this initial
budgeted experiment runs on CPU. No system packages or inference dependencies change.

```bash
uv venv --python 3.12 training/.venv
uv pip install --python training/.venv/bin/python --no-cache \
  torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python training/.venv/bin/python --no-cache -r training/requirements.txt
HF_HUB_DISABLE_XET=1 training/.venv/bin/python training/embedding_experiment.py \
  --dataset evaluation/artifacts/training-expanded-v3 \
  --references data/generated/expanded/training_references.jsonl \
  --images-dir data/generated/expanded/images \
  --output training/artifacts/dinov2-small \
  --threads 6 --batch-size 16 --epochs 12
```

`requirements.lock.txt` records the complete installed environment. It contains the
CPU torch version; restoring it requires the CPU wheel index for that package.
Downloads and generated features are ignored by Git, under `training/artifacts/`.

## Data and safeguards

- Preserve the existing producer/class-isolated splits in `queries.jsonl`.
- Encode every validation query and training variants 0 and 4 (one single-bottle and
  one centred three-bottle view per source). Do not decode, embed, score, or train on
  any test query. The full reference gallery can include held-out identities, as
  required for open-set instance retrieval; held-out references never enter the loss.
- Composite transparency on white and apply the same centre-column prior to wide
  gallery and query photos. Resize the short edge to 256 and centre crop 224 pixels.
- Concatenate CLS and mean patch features, then L2-normalize. Train a linear
  identity-initialized projection with multi-positive contrastive loss. Same-slug
  alternate reference views are all positives, never false negatives.
- Pin model revision and record input manifest hashes. Refuse to reuse cached features
  if model, inputs or preprocessing differ. Fix the random seed to `20260921`.
- Select a trained checkpoint only if validation Accuracy@1 exceeds the baseline.
  Do not treat the final training epoch as automatically better.

## Outputs

`baseline.json` and `baseline_predictions.jsonl` appear as soon as frozen validation
retrieval is measured. `features.npz` caches gallery, validation, and training
features, and `features.json` identifies their exact inputs. `report.json` records
all epoch losses and validation measurements, the selected checkpoint and test-query
usage (always zero). A winning trained head is saved as `projection.pt`, with its
manifest/model fingerprint. `selected_predictions.jsonl` describes that head's
validation predictions.

These validation images are synthetic derivatives of catalogue photos. Results
measure controlled retrieval robustness, not independent real-phone-photo accuracy
and not a guarantee of a competition result. A separate held-out real-photo test is
still required before making production-quality accuracy claims.

## Export the selected fallback

When the trained projection beats the frozen validation baseline, export the selected
backbone and projection to ONNX together with its projected reference gallery:

```bash
training/.venv/bin/python training/export_onnx.py
```

The exporter works offline from the pinned checkpoint cache, verifies the ONNX model,
and records checksums for the model, gallery, projection, catalogue slugs and reference
manifest under `data/generated/expanded/dinov2/`. The runtime can therefore load the
neural fallback without PyTorch or the training environment.
