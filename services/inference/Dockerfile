FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CIFR_CATALOG_PATH=/app/data/catalog.jsonl \
    CIFR_VISUAL_INDEX_DIR=/app/data/siglip \
    CIFR_LOCAL_FEATURE_INDEX=/app/data/local_features.npz \
    CIFR_OCR_MODEL_DIR=/app/data/ocr \
    CIFR_FUSION_PATH=/app/data/fusion.json \
    CIFR_IMAGES_DIR=/app/data/images \
    CIFR_NEURAL_ENABLED=0

WORKDIR /app
COPY services/inference/requirements.txt services/inference/requirements-ocr.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps -r requirements-ocr.txt

COPY services/inference/app ./app
COPY services/inference/data/ocr ./data/ocr
COPY data/generated/expanded/catalog.jsonl ./data/catalog.jsonl
COPY data/generated/expanded/siglip/model.onnx data/generated/expanded/siglip/encoder.json \
     data/generated/expanded/siglip/gallery.npz data/generated/expanded/siglip/gallery.json ./data/siglip/
COPY data/generated/expanded/local_features.npz data/generated/expanded/local_features.npz.sha256 ./data/
COPY services/inference/data/fusion.json ./data/fusion.json
COPY services/inference/data/adapter.npy services/inference/data/adapter.json ./data/siglip/
COPY data/generated/expanded/images ./data/images

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health/ready')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
