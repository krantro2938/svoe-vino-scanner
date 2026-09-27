from __future__ import annotations

import io
import json

import httpx
import pytest
from PIL import Image

from app.config import Settings
from app.features import DESCRIPTOR_VERSION, extract_descriptors
from app.imaging import center_wine_view, decode_image, prepare_rgb
from app.main import create_app
from app.ocr import LabelOCR, OCRMatch, _tokens
from app.retrieval import _ocr_confidence


def image_bytes(color: tuple[int, int, int], format: str = "PNG", size=(128, 160)) -> bytes:
    stream = io.BytesIO()
    image = Image.new("RGB", size, color)
    for y in range(30, 130, 8):
        for x in range(24, 104, 8):
            image.putpixel((x, y), (255 - color[0], 255 - color[1], 255 - color[2]))
    image.save(stream, format=format)
    return stream.getvalue()


def test_transparent_index_and_query_share_white_background():
    source = Image.new("RGBA", (48, 64), (0, 0, 0, 0))
    for y in range(12, 52):
        for x in range(16, 32):
            source.putpixel((x, y), (120, 20, 35, 255))
    stream = io.BytesIO()
    source.save(stream, format="PNG")
    expected = prepare_rgb(source)
    decoded = decode_image(stream.getvalue(), max_pixels=10_000, min_side=32)
    assert decoded.image.mode == "RGB"
    assert list(decoded.image.getdata()) == list(expected.getdata())


def test_ocr_transliterates_and_ranks_exact_terms():
    tokens = _tokens("ТАБИЯ Нуар ARISTOV DONUM XXIV")
    assert {"tabiya", "нуар", "nuar", "donum", "24"} <= tokens

    reader = LabelOCR.__new__(LabelOCR)
    reader.fields = {
        "target": [({"aristov", "donum", "24"}, 2.0)],
        "other": [({"aristov", "rose"}, 2.0)],
    }
    reader.idf = {"aristov": 1.0, "donum": 4.0, "24": 4.0, "rose": 4.0}
    match = reader.rank_text("ARISTOV DONLM X1V")
    assert match is not None
    assert match.slug == "target"
    assert match.margin > 0


def test_ocr_ignores_generic_colour_words_in_noisy_label_text():
    reader = LabelOCR.__new__(LabelOCR)
    reader.fields = {
        "massandra-white": [({"muskatel"}, 2.0)],
        "unrelated-red-blend": [({"red", "ред", "blend"}, 2.0)],
    }
    reader.idf = {"muskatel": 4.0, "red": 8.0, "ред": 8.0, "blend": 8.0}
    match = reader.rank_text("МУСКАТЕДЛ ВИНО РЕД")
    assert match is not None
    assert match.slug == "massandra-white"
    assert match.candidates[0].matched_tokens == ("muskatel",)


def test_ocr_exposes_candidates_and_flags_producer_and_vintage_conflicts():
    reader = LabelOCR.__new__(LabelOCR)
    reader.fields = {
        "target": [({"distinctive"}, 2.0), ({"maker"}, 1.5)],
        "producer-only": [({"ordinary"}, 2.0), ({"rival"}, 1.5)],
    }
    reader.idf = {"distinctive": 5.0, "maker": 3.0, "rival": 3.0, "2024": 1.0}
    reader.producer_tokens = {"maker", "rival"}
    reader.winery_tokens = {"target": {"maker"}, "producer-only": {"rival"}}
    reader.candidate_years = {"target": {"2022"}, "producer-only": set()}
    reader.alias_tokens = {"target": set(), "producer-only": set()}
    reader.canonical_tokens = {
        "target": {"distinctive", "maker", "2022"},
        "producer-only": {"ordinary", "rival"},
    }

    match = reader.rank_text("distinctive rival 2024")

    assert match is not None
    assert match.slug == "target"
    assert match.producer_conflict is True
    assert match.vintage_conflict is True
    assert [candidate.slug for candidate in match.candidates] == ["target", "producer-only"]
    assert match.candidates[0].matched_tokens == ("distinctive",)


def test_ocr_confidence_caps_conflicts_alias_dependency_and_weak_margin():
    strong = dict(slug="wine", score=100.0, margin=100.0, text="label")

    assert 0.95 < _ocr_confidence(OCRMatch(**strong), visually_indexed=True) <= 0.96
    assert _ocr_confidence(
        OCRMatch(**strong, producer_conflict=True), visually_indexed=True
    ) == pytest.approx(0.55)
    assert _ocr_confidence(
        OCRMatch(**strong, vintage_conflict=True), visually_indexed=True
    ) == pytest.approx(0.55)
    assert _ocr_confidence(
        OCRMatch(**strong, alias_dependent=True), visually_indexed=True
    ) == pytest.approx(0.69)
    assert _ocr_confidence(OCRMatch(**strong), visually_indexed=False) == pytest.approx(0.69)
    assert _ocr_confidence(
        OCRMatch(slug="wine", score=100.0, margin=0.0, text="label"),
        visually_indexed=True,
    ) == pytest.approx(0.75)


def test_ocr_marks_a_match_that_depends_on_noncanonical_alias_tokens():
    reader = LabelOCR.__new__(LabelOCR)
    reader.fields = {"target": [({"tabia", "pinot", "2025"}, 2.0)]}
    reader.idf = {"tabia": 4.0, "pinot": 2.0, "2025": 2.0}
    reader.producer_tokens = set()
    reader.winery_tokens = {"target": {"melnikov"}}
    reader.candidate_years = {"target": {"2025"}}
    reader.alias_tokens = {"target": {"tabia", "pinot", "2025"}}
    reader.canonical_tokens = {"target": {"pinot", "2025", "melnikov"}}

    match = reader.rank_text("Tabia Pinot 2025")

    assert match is not None
    assert match.alias_dependent is True
    assert match.vintage_conflict is False


def test_ocr_fast_pass_skips_remaining_regions_when_evidence_is_sufficient():
    reader = LabelOCR.__new__(LabelOCR)
    reader.workers = 1
    reader.fields = {"target": [({"distinctive"}, 2.0)]}
    reader.idf = {"distinctive": 5.0}
    reader.producer_tokens = set()
    reader.winery_tokens = {"target": set()}
    reader.candidate_years = {"target": set()}
    reader.alias_tokens = {"target": set()}
    reader.canonical_tokens = {"target": {"distinctive"}}
    calls = []

    def fake_read_regions(prepared, directory, indexes):
        calls.append(indexes)
        return "distinctive"

    reader._read_regions = fake_read_regions
    match = reader.rank(Image.new("RGB", (64, 64), "white"), minimum_score=7.0)

    assert match is not None and match.slug == "target"
    assert calls == [(2, 3, 6)]


def test_center_view_preserves_portrait_and_excludes_side_bottles():
    portrait = Image.new("RGB", (128, 200), "red")
    assert center_wine_view(portrait) is portrait
    scene = Image.new("RGB", (600, 300), "white")
    scene.paste("red", (190, 0, 410, 300))
    target = center_wine_view(scene)
    assert target.size == (78, 300)
    assert target.getextrema() == ((255, 255), (0, 0), (0, 0))


def test_ocr_keeps_complete_center_words_and_rejects_easier_side_label():
    tsv = "level\tleft\twidth\tconf\ttext\n"
    tsv += "5\t0\t20\t99\tneighbor\n"
    tsv += "5\t20\t60\t80\tcentral\n"
    tsv += "5\t80\t20\t99\tdistinctive\n"
    assert LabelOCR._central_words(tsv, 30, 70) == "central"
    # If the middle is unreadable, never substitute neighboring label text.
    assert LabelOCR._central_words(tsv, 60, 70) == ""


def test_ocr_requests_tsv_without_relying_on_system_config_files(monkeypatch, tmp_path):
    from types import SimpleNamespace

    reader = LabelOCR.__new__(LabelOCR)
    reader.executable = "tesseract"
    reader.tessdata_dir = tmp_path
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=(
            "level\tleft\twidth\tconf\ttext\n"
            "5\t40\t20\t90\tcentral\n"
            "5\t0\t20\t99\tneighbor\n"
        ))

    monkeypatch.setattr("app.ocr.subprocess.run", run)
    assert reader._read_one((tmp_path / "image.png", 6, "eng", 30, 70)) == "central"
    assert commands[0][-2:] == ["-c", "tessedit_create_tsv=1"]


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.fixture()
async def client(tmp_path):
    catalog = [
        {
            "slug": "red-wine",
            "name": "Красное тестовое",
            "winery": "Тестовая винодельня",
            "category": "Красное",
            "region": "Кубань",
            "grapes": "Каберне Совиньон",
        },
        {
            "slug": "white-wine",
            "name": "Белое тестовое",
            "winery": "Тестовая винодельня",
            "category": "Белое",
            "region": "Крым",
            "grapes": "Рислинг",
        },
    ]
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")

    manifest_items = []
    for slug, color in (("red-wine", (190, 25, 35)), ("white-wine", (235, 225, 150))):
        with Image.open(io.BytesIO(image_bytes(color))) as image:
            descriptors = [value.tolist() for value in extract_descriptors(image)]
        manifest_items.append({"slug": slug, "descriptors": descriptors})
    manifest_path = tmp_path / "index.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model_version": "test-index-v1",
                "descriptor_version": DESCRIPTOR_VERSION,
                "items": manifest_items,
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        catalog_path=catalog_path,
        index_manifest_path=manifest_path,
        max_upload_bytes=200_000,
        max_image_pixels=1_000_000,
        min_image_side=32,
        found_threshold=0.6,
    )
    application = create_app(settings)
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://test"
        ) as test_client:
            yield test_client


@pytest.mark.anyio
async def test_health_and_catalog(client):
    assert (await client.get("/health/live")).json() == {"status": "ok", "catalog_count": None, "indexed_count": None, "model_version": None}
    ready = await client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["catalog_count"] == 2
    assert ready.json()["indexed_count"] == 2
    assert (await client.get("/v1/wines/red-wine")).json()["grapes"] == "Каберне Совиньон"
    assert (await client.get("/v1/wines/missing")).status_code == 404


@pytest.mark.anyio
async def test_local_frontend_cors(client):
    for origin in ("http://localhost:3003", "http://localhost:3004"):
        response = await client.options(
            "/v1/search",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin


@pytest.mark.parametrize("format", ["JPEG", "PNG", "WEBP"])
@pytest.mark.anyio
async def test_eval_decodes_by_content_not_extension(client, format):
    payload = image_bytes((190, 25, 35), format=format)
    response = await client.post(
        "/v1/eval/predict",
        files={"image": ("deliberately-wrong.gif", payload, "application/octet-stream")},
    )
    assert response.status_code == 200
    assert response.json() == {"slug": "red-wine"}
    assert "X-Process-Time-Ms" in response.headers


@pytest.mark.anyio
async def test_search_returns_rich_contract(client):
    response = await client.post(
        "/v1/search", files={"image": ("query.bin", image_bytes((235, 225, 150)))}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["wine"]["slug"] == "white-wine"
    assert body["model_version"] == "test-index-v1"
    assert 0 <= body["confidence"] <= 1
    assert body["latency_ms"] >= 0
    assert body["status"] == "found"


@pytest.mark.anyio
async def test_multiple_bottles_select_center_despite_larger_side_area(client):
    scene = Image.new("RGB", (384, 160))
    side = Image.open(io.BytesIO(image_bytes((235, 225, 150))))
    center = Image.open(io.BytesIO(image_bytes((190, 25, 35))))
    scene.paste(side, (0, 0))
    scene.paste(center, (128, 0))
    scene.paste(side, (256, 0))
    stream = io.BytesIO()
    scene.save(stream, format="PNG")
    response = await client.post(
        "/v1/eval/predict", files={"image": ("three-bottles.png", stream.getvalue())}
    )
    assert response.status_code == 200
    assert response.json() == {"slug": "red-wine"}


@pytest.mark.anyio
async def test_input_errors_are_small_and_structured(client):
    missing = await client.post("/v1/eval/predict")
    assert missing.status_code == 422
    assert missing.json()["error"]["code"] == "image_required"

    malformed = await client.post(
        "/v1/eval/predict", files={"image": ("image.jpg", b"not an image")}
    )
    assert malformed.status_code == 400
    assert malformed.json()["error"]["code"] == "invalid_image"

    oversized = await client.post(
        "/v1/eval/predict", files={"image": ("image.bin", b"x" * 200_001)}
    )
    assert oversized.status_code == 413
    assert oversized.json()["error"]["code"] == "payload_too_large"


@pytest.mark.anyio
async def test_pairing_is_deterministic_and_catalog_grounded(client):
    request = {"slug": "white-wine", "dish": "лосось", "sauce": "лимонный"}
    first = await client.post("/v1/sommelier/pairing", json=request)
    second = await client.post("/v1/sommelier/pairing", json=request)
    assert first.status_code == 200
    assert first.json() == second.json()
    assert first.json()["verdict"] == "excellent"
    assert first.json()["grounded_in"]["grapes"] == "Рислинг"
    assert (await client.post(
        "/v1/sommelier/pairing", json={"slug": "missing", "dish": "рыба"}
    )).status_code == 404
