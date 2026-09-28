import numpy as np

from app.analogs import AnalogFinder
from app.label_text import TextMatcher, Word, normalize_tokens, style_of
from app.models import PairingRequest, Wine
from app.pipeline import FEATURES, FusionModel, candidate_features, shortlist
from app.sommelier import recommend


def wine(slug, name, winery="Винодельня", category="Белое", grapes=None, region="Крым"):
    return Wine(slug=slug, name=name, winery=winery, category=category, grapes=grapes, region=region)


CATALOG = {
    w.slug: w
    for w in (
        wine("katharon-risling-suhoe", "Катарон Рислинг", "Катарон", grapes="Рислинг"),
        wine("katharon-shardone", "Катарон Шардоне", "Катарон", grapes="Шардоне"),
        wine("massandra-muskatel-belyy-sladkoe-16", "Мускатель белый", "Массандра", grapes="Мускат"),
        wine("massandra-portveyn-belyy-sladkoe", "Портвейн белый", "Массандра", grapes="Кокур"),
        wine("abrau-bryut", "Абрау брют", "Абрау-Дюрсо", grapes="Шардоне"),
        wine("other-red-suhoe", "Каберне", "Другая", "Красное", grapes="Каберне Совиньон", region="Кубань"),
    )
}


def test_tokens_fix_homoglyphs_camel_case_and_roman_numerals():
    # OCR reads "МАССАНДРА" as Latin look-alikes and drops the Д.
    assert "массанара" in normalize_tokens("MACCAHAPA")
    assert {"пино", "нуар", "pino", "nuar"} <= set(normalize_tokens("ПиноНуар"))
    assert "24" in normalize_tokens("XXIV")
    matcher = TextMatcher(CATALOG)   # the fuzzy step restores the brand
    assert matcher._resolve("массанара") == [("массандра", 0.6)]


def test_label_text_normalisation_handles_real_ocr_failure_modes():
    assert {"красное", "2022"} <= set(normalize_tokens("KPACHOE2022"))      # glued digits + homoglyphs
    assert "аратти" in normalize_tokens("АРАТTИ")                            # Latin T inside Cyrillic
    assert {"каберне", "фран"} <= set(normalize_tokens("CABERNET FRANC"))  # Latin label, Russian catalogue
    matcher = TextMatcher(CATALOG)
    assert matcher._resolve("катаронрислинг") == [("катарон", 0.8), ("рислинг", 0.8)]


def test_greek_lookalikes_and_weighted_style_vote():
    assert "алушта" in normalize_tokens("ΑЛΥШТА")          # Greek Α and Υ from the recogniser
    assert style_of("CYXOE KPACHOE") == {"sweetness": "dry", "colour": "red"}
    from app.label_text import query_style
    words = [Word("blanc", 0.9, 0.25), Word("sélection rouge", 0.9, 0.9)]   # neighbour at the edge
    assert query_style(words)["colour"] == "red"


def test_style_parser_prefers_specific_sweetness_phrases():
    assert style_of("Игристое экстра брют")["sweetness"] == "extra_brut"
    assert style_of("вино полусухое розовое") == {"sweetness": "semi_dry", "colour": "rose"}


def test_text_matcher_scores_words_once_and_flags_style_conflicts():
    matcher = TextMatcher(CATALOG)
    evidence = matcher.score([Word("МАССАНДРА", 0.9, 1.0), Word("МУСКАТЕЛЬ", 0.9, 1.0), Word("сухое", 0.9, 1.0)])
    best = max(evidence, key=lambda slug: evidence[slug].score)
    assert best == "massandra-muskatel-belyy-sladkoe-16"
    assert evidence[best].style_conflict  # label says dry, catalogue says sweet
    # Cyrillic and transliterated variants of one OCR word are counted once.
    assert evidence[best].matched.count("массандра") == 1


def test_fuzzy_ocr_tokens_match_nearest_catalogue_token():
    matcher = TextMatcher(CATALOG)
    evidence = matcher.score([Word("КАТАРОН", 0.9, 1.0), Word("РИСЛИНГГ", 0.8, 1.0)])
    assert max(evidence, key=lambda s: evidence[s].score) == "katharon-risling-suhoe"


def test_explained_away_feature_penalises_sibling_missing_a_label_word():
    matcher = TextMatcher(CATALOG)
    text = matcher.score([Word("Катарон", 0.9, 1.0), Word("Рислинг", 0.9, 1.0)])
    slugs = ["katharon-shardone", "katharon-risling-suhoe"]
    index_of = {s: i for i, s in enumerate(slugs)}
    visual = {name: np.array([0.9, 0.9]) for name in ("mean", "max", "full>bottle", "centre>label")}
    rows = candidate_features(slugs, visual, index_of, text, {}, matcher.idf)
    assert set(rows[0]) == set(FEATURES)
    assert rows[0]["t_missing"] > 0 and rows[1]["t_missing"] == 0


def test_fusion_probabilities_are_normalised_and_ordered():
    model = FusionModel({"v_mean": 10.0, "sift_inliers": 1.0}, not_found_threshold=0.3)
    probabilities = model.probabilities([{"v_mean": 0.9, "sift_inliers": 3.0}, {"v_mean": 0.8, "sift_inliers": 0.0}])
    assert np.isclose(probabilities.sum(), 1.0) and probabilities[0] > probabilities[1]
    assert model.not_found_threshold == 0.3


def test_shortlist_unions_visual_and_text_candidates():
    visual = {"mean": np.array([0.2, 0.9, 0.1])}
    matcher = TextMatcher(CATALOG)
    text = matcher.score([Word("Абрау", 0.9, 1.0)])
    chosen = shortlist(visual, ["a", "b", "abrau-bryut"], text)
    assert chosen[0] == "b" and "abrau-bryut" in chosen


def test_analogs_come_from_other_wineries_with_reasons():
    finder = AnalogFinder(CATALOG)
    analogs = finder.similar("katharon-shardone", limit=3)
    assert analogs and all(w.winery != "Катарон" for w, _ in analogs)
    assert analogs[0][0].slug == "abrau-bryut" and any("Шардоне".casefold() in r for r in analogs[0][1])


def test_sommelier_offers_better_fitting_catalogue_wines():
    response = recommend(PairingRequest(slug="other-red-suhoe", dish="fish"), CATALOG, AnalogFinder(CATALOG))
    assert response.verdict == "not_ideal"
    assert response.alternatives and all(a.slug != "other-red-suhoe" for a in response.alternatives)
    assert response.glass and response.serving_temperature
