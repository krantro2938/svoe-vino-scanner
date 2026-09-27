# Catalog ingestion

`build_catalog.py` turns the supplied CSV and multipart Strapi RAR into small,
deterministic manifests. It lists archive members with 7-Zip and does not extract
the archive.

Run from any directory:

```bash
python3 scripts/build_catalog.py
```

Outputs are written to `data/generated/catalog/`:

- `catalog.jsonl`: exact-deduplicated, whitespace-cleaned catalog rows with the
  selected archive path and CRC32 when resolved, plus explicit `index_ready`
  gates and blockers;
- `media_manifest.jsonl`: one reconciliation decision per unique CSV photo name;
- `catalog_import_report.json`: source SHA-256 values, missing-field and conflict
  audits, and all unresolved/ambiguous photo names;
- `archive_inventory.json`: compact counts for the unextracted archive.

Audit the generated records and reviewed public-label aliases without changing
or guessing any mapping:

```bash
python3 scripts/audit_catalog.py --output /tmp/cifr-catalog-audit.json
```

The audit independently recomputes index blockers, expands every shared-photo
group to its affected identities, detects stale generated metadata, and reports
alias tokens missing from the catalogue identity. The current catalogue has
2,032 ready records and 71 blocked records; 13 shared-photo groups affect 26
records. A non-empty alias-gap report is evidence for catalogue coverage review,
not permission to rewrite source identities.

The importer first compares transliterated normalized names, then permits a
separator-insensitive comparison only when it identifies one canonical filename
family. It always prefers a full-size original over Strapi renditions. Multiple
separately uploaded originals sharing the same source name remain `ambiguous`—a
size-based selection is recorded for inspection but is not treated as proven.

To resolve a reviewed exception, add an exact mapping from the CSV photo name to
the full archive member path in `data/catalog/media_overrides.json`, then rebuild.
Overrides are validated against the archive listing. Use `--skip-source-hashes`
only for quick iteration; release manifests should contain the source hashes.

Unresolved rows include fuzzy, review-only suggestions. They are deliberately not
auto-selected: filename similarity cannot establish that two bottle images depict
the same catalog item. Same-name reuploads are considered equivalent only when
both their RAR CRC32 and byte size agree.

Ambiguous rows keep a `media_review_candidate_path`, but `media_path` stays null
until an override is reviewed. Rows whose photo is shared by multiple slugs are
also kept out of the index-ready set even when the archive filename itself is
resolved.

## Selective extraction

Preview the exact extraction allowlist without writing anything or invoking
7-Zip:

```bash
python3 scripts/extract_catalog_media.py --dry-run
```

Smoke-test two files in a temporary location:

```bash
python3 scripts/extract_catalog_media.py \
  --limit 2 \
  --output-dir /tmp/cifr-catalog-images \
  --index-manifest /tmp/cifr-extracted-media.jsonl
```

Extract every resolved original for the inference index:

```bash
python3 scripts/extract_catalog_media.py
```

The extractor gives each local file its exact catalog `photo_name`, even when
Strapi renamed the archived upload. It rejects path separators, traversal,
control characters, overlong names, malformed manifest rows, and mismatched
size/CRC values. It passes a generated allowlist to one `7z x` invocation, so it
never expands unrelated uploads. Existing verified files are reused. Conflicting
existing bytes cause a failure unless `--overwrite` is supplied; replacements
are verified and installed atomically.

`data/generated/catalog/extracted_media.jsonl` maps catalog names and slugs to
the local filenames. Its `index_ready` flag is false for a resolved photo shared
by multiple slugs. The current inference indexer can consume the files directly:

```bash
cd services/inference
PYTHONPATH=. .venv/bin/python tools/build_index.py \
  --catalog ../../data/generated/catalog/catalog.jsonl \
  --images-dir ../../data/catalog-images \
  --output data/index_manifest.json
```
