# Сканер российских вин — «Своё Вино»

Покупатель фотографирует этикетку у полки — сервис за ~1,5 секунды открывает **одну**
карточку вина из каталога «Своё Вино»: производитель, регион, сорт, описание, подача,
цифровой сомелье и аналоги из других виноделен. Если вина в каталоге нет, сервис честно
говорит об этом и показывает похожие вина, а не выдаёт чужую карточку за найденную.

## Результаты

Field-like набор field-v1: 700 фото, 700 разных вин; ранкер обучен на отдельном field-v2.
Прогон через тот же движок, что и API, с живым OCR, на CPU ноутбука (12 потоков, без GPU):

| Метрика | Прежний пайплайн | Сейчас |
| --- | ---: | ---: |
| Accuracy@1 = micro-F1 (top-1) | 0,660 | **0,931** |
| Macro-F1 (top-1) | — | **0,887** |
| Recall@5 (верное вино в top-5) | 0,660¹ | **0,996** |
| MRR@5 | — | **0,959** |
| Точность ответов с уверенностью ≥ 0,80 (≈83% фото, показываются сразу одной карточкой) | — | **0,988** |
| Задержка p50 / p95 | 0,29 / 1,30 с | **1,13 / 1,33 с** |

На винах, которых ранкер не видел при обучении (оценка по признакам), — 0,931 / 0,990 (Top-1 / Top-5).
¹ Прежний пайплайн возвращал одного кандидата, поэтому его Recall@5 равен Top-1.
Отчёт evaluator-а: [docs/FIELD_BENCHMARK_REPORT.md](docs/FIELD_BENCHMARK_REPORT.md) (статус VALID, 0 null).

Как измерено: публичный набор организатора — 3 фото без ответов, причём два из них
(Табия «Пино Нуар» с шишкой и Aristov DONUM XXIV) отсутствуют и в CSV, и на сайте.
Поэтому точность измеряется на **field-like наборе** (`evaluation/field_benchmark.py`):
эталон каждого вина рендерится как фото у полки — изгиб этикетки на бутылке,
перспектива, соседние бутылки по краям, блики, баланс белого, размытие, шум, JPEG.
Fusion обучается на field-v2 (1 500 фото), оценка — на отдельном field-v1 (700 фото).
Это синтетика, а не приватный набор организатора; реальные фото могут быть сложнее.

## Как это работает

```text
фото ─► нормализация ─► SigLIP 2 (2 вида кадра × 2 вида эталона) ─► shortlist top-30 ┐
   └──► PP-OCRv5 (кириллица+латиница) ─► текстовый матчинг по каталогу ─► top-10 ────┤
                                                                                      ▼
                   RootSIFT + RANSAC по shortlist ─► 18 признаков ─► обученный softmax-ранкер
                                                                                      │
             {"slug": "…"}  ·  top-5 с вероятностями  ·  found / uncertain / not_found
```

Подробно — в [ARCHITECTURE.md](./ARCHITECTURE.md).

## Быстрый старт

### Docker (после сборки артефактов, см. ниже)

```bash
docker compose up --build
```

UI — `http://localhost:3003`, API — `http://localhost:8081`. С телефона в той же сети
откройте `http://<IP-компьютера>:3003` (CORS разрешает локальные сети).

### Локально

```bash
# 1. API (Python 3.13+)
cd services/inference
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-deps -r requirements-ocr.txt
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8080

# 2. UI
cd apps/web && npm ci
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8080 npm run dev -- --host 0.0.0.0
```

### Артефакты моделей

Индексы (~760 МБ: SigLIP ONNX и галерея, RootSIFT, эталонные фото) не хранятся в git.
Скачайте их из GitHub Release:

```bash
scripts/download_artifacts.sh      # → data/generated/expanded/
```

### Сборка артефактов из `Датасет/`

Пакет организатора (CSV каталога, multipart RAR с фото, `eval/`) в репозиторий не входит —
положите его в `Датасет/` в корне репозитория.

```bash
uv venv --python 3.12 training/.venv
uv pip install --python training/.venv/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python training/.venv/bin/python -r training/requirements.txt onnx
scripts/build_artifacts.sh          # каталог → медиа → снапшот сайта → SigLIP ONNX → галерея → RootSIFT
```

Скрипт использует снапшот `data/external/vino-svoe-20260927`; новый снапшот делается
`scripts/scrape_wine_catalog.py --output data/external/vino-svoe-<дата>` (каталог растёт
~50 вин в день). Веса ранкера лежат в репозитории (`services/inference/data/fusion.json`),
модели OCR — в `services/inference/data/ocr/`.

### Скрипт организатора

```bash
cd Датасет/eval
./participant_test.sh --images-dir ./queries --manifest ./queries.tsv \
  --endpoint http://127.0.0.1:8080/v1/eval/predict --output ./predictions.jsonl
```

## API

| Метод | Путь | Назначение |
| --- | --- | --- |
| POST | `/v1/eval/predict` | `{"slug": "..."}` для скрипта организатора (всегда отвечает slug) |
| POST | `/v1/search` | карточка, `status`, `confidence` (top-1), `confidence_top5`, `top1_top2_margin`, `top5`, `analogs`, тайминги стадий |
| GET | `/v1/wines/{slug}` | карточка каталога |
| GET | `/v1/wines/{slug}/image` | эталонное фото |
| GET | `/v1/wines/{slug}/analogs` | аналоги других виноделен с причинами |
| POST | `/v1/sommelier/pairing` | `{slug, dish, occasion?, preference?}` → вердикт, подача, бокал, альтернативы |
| GET | `/health/ready` | готовность, размер каталога и индекса, версия модели |

## Переменные окружения

| Переменная | По умолчанию | Смысл |
| --- | --- | --- |
| `CIFR_CATALOG_PATH` | `data/generated/expanded/catalog.jsonl` | каталог |
| `CIFR_VISUAL_INDEX_DIR` | `data/generated/expanded/siglip` | ONNX-энкодер и галерея |
| `CIFR_LOCAL_FEATURE_INDEX` | `data/generated/expanded/local_features.npz` | RootSIFT |
| `CIFR_OCR_MODEL_DIR` | `services/inference/data/ocr` | PP-OCRv5 |
| `CIFR_FUSION_PATH` | `services/inference/data/fusion.json` | веса ранкера и порог «нет в каталоге» |
| `CIFR_IMAGES_DIR` | `data/generated/expanded/images` | эталонные фото для карточек |
| `CIFR_FOUND_PROBABILITY` | `0.80` | порог «найдено» без экрана выбора |
| `CIFR_VISUAL_ENABLED`, `CIFR_OCR_ENABLED` | `1` | выключение каналов для экспериментов |

## Проверки

```bash
cd services/inference && .venv/bin/python -m pytest
cd apps/web && npm run lint && npm run build
python3 -m unittest discover -s tests/catalog && python3 -m unittest discover -s tests/evaluation
```

Воспроизвести метрики: [evaluation/README.md](./evaluation/README.md).

## Ограничения

- Метрики получены на синтетических field-like фото; приватный набор организатора может быть сложнее.
- Одна серия с одинаковой этикеткой и разным винтажом/сладостью различается только по тексту;
  если год или «полусладкое» на фото не читается, сервис отвечает `uncertain` и показывает варианты.
- В CSV нет рейтинга Роскачества и крепости — UI их не выдумывает.
- Работает на CPU; GPU не требуется. На RTX 4050 можно ускорить SigLIP через onnxruntime-gpu.
