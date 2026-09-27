# Public catalogue snapshot

`python scripts/scrape_wine_catalog.py` reads the public `https://vino-svoe.ru/wines`
HTML pages, decodes their embedded Nuxt SSR data and downloads the linked 910 px
bottle image rendition. It does not call the site's private or `/api/` endpoints.
The site's robots policy explicitly permits pagination with `page=`; the image
host permits `*/img/*`. A saved robots copy accompanies the snapshot. The script stops if that allowance changes.

The default output is `data/external/vino-svoe/`, separate from the official
competition catalogue. Run again to resume cached pages/images; use a new output
directory for a fresh dated snapshot. Default global request spacing is 0.5 seconds
with four workers and retry backoff. `--skip-images` captures metadata alone.

Outputs:

- `catalog.jsonl`: one record per exact site slug, with name/category/color/region/
  winery, source page and wine URL, observed timestamp, image URL, and image hashes.
- `pages/`: original public HTML responses, retained for provenance and re-parsing.
- `images/`: public 910 px renditions or CRC-verified organizer originals matching
  the exact public upload; each record identifies which source was used.
- `report.json`: expected/actual pages and wines, duplicate slugs, image failures,
  completion flag and license status. Completion refers to catalogue enumeration;
  image count/failures must also be inspected before training.

`image_path` is relative to the snapshot directory; `media_path` is relative to
where the scraper was run. `image_sha256` and `media_sha256` are identical byte
hashes. `index_ready` becomes true only after an image is downloaded or a verified
organizer original is reused, and its file signature is checked. Missing and failed media remain explicit and are not silently
replaced with another wine. Listing metadata does not contain grapes or a full
description; absent fields are not inferred from names.

The site is mutable: repeated listings can omit or duplicate products if ordering
changes during capture. The report checks unique counts against the reported site
total. A complete snapshot is not a guarantee of a complete historical catalogue,
verified wine identity, or recognition accuracy. Preserve official competition
slugs during merges. Shared image hashes across different slugs need review before
training or evaluating recognition. Public availability does not establish a
redistribution or model-training license; provenance is retained for that review.

Existing organizer originals are reused only when the exact wine slug and upload
basename match a unique extraction manifest entry, and size plus CRC32 verify.
These records have `media_status=reused_organizer`, original archive provenance,
and a stored-file SHA256. The report counts downloaded and reused images separately.

The September 20, 2026 snapshot completed all 130 pages and 2,069 unique wine
records: 107 downloaded public renditions and 1,962 verified organizer originals,
with no duplicate listing slugs or failed images. A separate verification decoded
all 2,069 image files with Pillow and checked both stored SHA256 fields against
the actual bytes. This measures catalogue and media integrity, not recognition accuracy.
