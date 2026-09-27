"""Grounded "digital sommelier": guided pairing verdict and better-fitting analogues.

Every statement is derived from catalogue fields (category, colour, grapes,
region, description) and a transparent style/dish compatibility table; nothing
is invented about ratings or awards. Alternatives are real catalogue wines that
fit the dish better *and* stay close to the scanned wine's style.
"""
from __future__ import annotations

from dataclasses import dataclass

from .analogs import AnalogFinder
from .label_text import style_of
from .models import PairingAlternative, PairingRequest, PairingResponse, Wine


@dataclass(frozen=True, slots=True)
class WineStyle:
    key: str
    serving: str
    glass: str


DISH_KEYWORDS = {
    "red_meat": ("beef", "steak", "lamb", "burger", "meat", "мяс", "говяд", "стейк", "баран", "бургер", "шашлык"),
    "poultry": ("chicken", "turkey", "duck", "poultry", "куриц", "индей", "утк", "птиц"),
    "fish": ("fish", "salmon", "tuna", "рыб", "лосос", "тун", "форел"),
    "seafood": ("seafood", "shrimp", "oyster", "морепр", "кревет", "устриц", "мид"),
    "dessert": ("dessert", "cake", "chocolate", "десерт", "торт", "шоколад", "фрукт"),
    "cheese": ("cheese", "сыр"),
    "vegetables": ("vegetable", "salad", "mushroom", "овощ", "салат", "гриб"),
    "spicy": ("spicy", "curry", "asian", "остр", "карри", "азиат"),
    "aperitif": ("aperitif", "snack", "аперитив", "закуск", "фуршет"),
}
DISH_LABELS = {
    "red_meat": "красное мясо", "poultry": "птица", "fish": "рыба", "seafood": "морепродукты",
    "dessert": "десерт", "cheese": "сыры", "vegetables": "овощи и грибы", "spicy": "острые блюда",
    "aperitif": "аперитив и закуски", "general": "ваше блюдо",
}
STYLE_SCORES = {
    "red_meat": {"red": 4, "rose": 2, "sparkling": 1, "white": 1, "sweet": 0},
    "poultry": {"red": 3, "white": 3, "rose": 3, "sparkling": 2, "sweet": 1},
    "fish": {"white": 4, "rose": 3, "sparkling": 3, "red": 1, "sweet": 0},
    "seafood": {"white": 4, "sparkling": 4, "rose": 2, "red": 0, "sweet": 0},
    "dessert": {"sweet": 4, "sparkling": 2, "rose": 1, "white": 1, "red": 0},
    "cheese": {"red": 3, "white": 3, "sparkling": 3, "sweet": 3, "rose": 2},
    "vegetables": {"white": 4, "rose": 3, "sparkling": 3, "red": 2, "sweet": 1},
    "spicy": {"sweet": 3, "rose": 3, "white": 3, "sparkling": 2, "red": 1},
    "aperitif": {"sparkling": 4, "white": 3, "rose": 3, "red": 1, "sweet": 1},
    "general": {"sparkling": 3, "rose": 3, "white": 3, "red": 3, "sweet": 2},
}
PAIR_WITH = {
    "red": ["стейк или мясо на гриле", "выдержанные твёрдые сыры", "блюда с грибами"],
    "white": ["белая рыба", "морепродукты", "салаты с зеленью"],
    "rose": ["птица", "овощи гриль", "мягкие сыры"],
    "sparkling": ["морепродукты и устрицы", "лёгкие закуски", "брют-канапе"],
    "sweet": ["фруктовые десерты", "голубые сыры", "орехи и фуа-гра"],
}
OCCASION_NOTES = {
    "dinner": "Для ужина вино откройте заранее и подайте к основному блюду.",
    "party": "Для компании удобно охладить сразу несколько бутылок и подать в широких бокалах.",
    "gift": "Для подарка выгодно смотрится, если рассказать о регионе и сорте — они указаны в карточке.",
    "date": "Для свидания подайте вино чуть прохладнее обычного — аромат раскроется в бокале.",
}
TASTE_STYLES = {"dry": {"dry", "brut", "extra_brut", "brut_nature"}, "sweet": {"semi_sweet", "sweet"},
                "semi": {"semi_dry", "semi_sweet"}}


def wine_style(wine: Wine) -> WineStyle:
    parsed = style_of(f"{wine.name} {wine.category or ''} {wine.slug.replace('-', ' ')}")
    text = f"{wine.name} {wine.category or ''} {wine.slug}".casefold()
    if any(word in text for word in ("игрист", "брют", "brut", "bryut", "петнат", "шампан")):
        return WineStyle("sparkling", "6–8 °C", "флюте или тюльпан")
    if parsed.get("sweetness") in {"sweet"} or "десерт" in text:
        return WineStyle("sweet", "10–12 °C", "небольшой десертный бокал")
    colour = parsed.get("colour") or ""
    if colour == "red" or "крас" in (wine.category or "").casefold():
        return WineStyle("red", "16–18 °C", "бордоский бокал")
    if colour == "rose" or "роз" in (wine.category or "").casefold():
        return WineStyle("rose", "8–10 °C", "бокал для белого вина")
    return WineStyle("white", "8–12 °C", "бокал для белого вина")


def _dish_key(request: PairingRequest) -> str:
    text = " ".join(filter(None, (request.dish, request.sauce))).casefold()
    for key, keywords in DISH_KEYWORDS.items():
        if key == text or any(keyword in text for keyword in keywords):
            return key
    return "general"


def _score(wine: Wine, dish_key: str) -> int:
    return STYLE_SCORES[dish_key].get(wine_style(wine).key, 1)


def _taste_ok(wine: Wine, preference: str | None) -> bool:
    wanted = TASTE_STYLES.get((preference or "").casefold())
    if not wanted:
        return True
    sweetness = style_of(f"{wine.name} {wine.category or ''} {wine.slug.replace('-', ' ')}").get("sweetness")
    return sweetness is None or sweetness in wanted


def recommend(request: PairingRequest, catalog: dict[str, Wine], analogs: AnalogFinder | None = None) -> PairingResponse:
    wine = catalog[request.slug]
    dish_key = _dish_key(request)
    style = wine_style(wine)
    score = _score(wine, dish_key)
    taste_ok = _taste_ok(wine, request.preference)
    verdict = {4: "excellent", 3: "good", 2: "acceptable"}.get(score, "not_ideal")
    if not taste_ok and verdict in {"excellent", "good"}:
        verdict = "acceptable"
    dish_label = DISH_LABELS[dish_key] if dish_key != "general" else f"«{request.dish}»"
    category = (wine.category or "вино").lower()
    grape_note = f" из сорта {wine.grapes}" if wine.grapes and "," not in wine.grapes else (
        f" из сортов {wine.grapes}" if wine.grapes else "")
    reasons = {
        "excellent": f"{category.capitalize()}{grape_note} — классическая пара для блюда: {dish_label}.",
        "good": f"{category.capitalize()}{grape_note} гармонично поддержит блюдо ({dish_label}).",
        "acceptable": f"С блюдом ({dish_label}) сочетание возможно, но вино и блюдо различаются по насыщенности.",
        "not_ideal": f"Для блюда ({dish_label}) это вино не лучший выбор — ниже варианты из каталога, которые подойдут точнее.",
    }
    explanation = reasons[verdict]
    if not taste_ok:
        explanation += " По сладости вино отличается от вашего предпочтения."
    if request.occasion in OCCASION_NOTES:
        explanation += " " + OCCASION_NOTES[request.occasion]

    alternatives: list[PairingAlternative] = []
    if analogs is not None:
        pool = analogs.similar(wine.slug, limit=60, other_wineries=False)
        better = [(w, r) for w, r in pool if _score(w, dish_key) > score and _taste_ok(w, request.preference)]
        same = [(w, r) for w, r in pool if _score(w, dish_key) >= max(score, 3) and _taste_ok(w, request.preference)
                and w.winery != wine.winery]
        for candidate, why in (better or same)[:3]:
            reason = ("Лучше подходит к блюду" if _score(candidate, dish_key) > score else "Похожий стиль от другой винодельни")
            alternatives.append(PairingAlternative(
                slug=candidate.slug, name=candidate.name, winery=candidate.winery,
                category=candidate.category, reason=f"{reason}: " + ", ".join(why[:2]),
            ))
    grounded = {"name": wine.name, "category": wine.category or ""}
    for key in ("grapes", "region", "winery"):
        value = getattr(wine, key)
        if value:
            grounded[key] = value
    return PairingResponse(
        slug=wine.slug,
        verdict=verdict,
        explanation=explanation,
        serving_temperature=style.serving,
        glass=style.glass,
        pair_with=PAIR_WITH[style.key],
        alternatives=alternatives,
        grounded_in=grounded,
        rules_version="pairing-rules-v2",
    )
