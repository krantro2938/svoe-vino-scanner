# Metadata confusion candidates

Run `python evaluation/build_confusion_clusters.py` after rebuilding the expanded
catalogue. Outputs live in `evaluation/artifacts/confusion_clusters/`:

- `clusters.jsonl`: overlapping groups with exact slugs, normalized winery tokens,
  grouping reason and deterministic cluster ID.
- `neighbors.jsonl`: every eligible slug, including those with no candidates, and
  up to 16 ordered candidate negatives with scores and reasons.
- `report.json`: source catalogue hash, parameters and coverage counts.

The September 21 build covers 1,389 of 2,070 eligible classes (67.10%) with 1,698
groups and 4,780 directed neighbor pairs. There are 85 equal-name-without-vintage
groups, 10 equal-slug-without-vintage groups, 122 series-token groups, 401 supplied
grape/category groups and 1,080 high-overlap name/slug pairs. Counts overlap;
these are candidate groups, not mutually exclusive classes.

All comparisons stay inside one normalized winery. Unicode normalization, case
folding and tokenization remove formatting differences. Only explicit four-digit
1900–2099 tokens are stripped as vintage candidates. Series signatures remove
provided grape/category/winery tokens and generic wine words. Near-name and
near-slug pairs require at least two shared informative tokens and token Jaccard
similarity of at least 0.65. Groups over 40 members are excluded and reported.
No connected-component expansion joins chains into a winery-wide group.

Priority scores are fixed grouping weights, not confidence or visual-similarity
probabilities. Equal name/slug signatures receive 1.0; series tokens 0.9; near
name 0.8; near slug 0.75; supplied grape/category 0.6. The highest applicable
weight wins; alphabetical exact-slug order breaks ties. Winery/name/grape fields
and labels are never rewritten, inferred or merged. Missing grapes remain missing.

For training, intersect both anchor and neighbor with the training split before
sampling; never move held-out examples into training. For inference, use these
neighbors to expand an existing retrieval shortlist, then score their actual
reference images/OCR evidence. This artifact by itself does not modify runtime
ranking and does not demonstrate model accuracy or actual visual confusion.
