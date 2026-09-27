# Offline OCR models

The recognizer uses Tesseract's official `eng`, `rus`, and `rus_best` trained-data
files for offline label reading. The Russian files come from the upstream
[`tessdata_fast`](https://github.com/tesseract-ocr/tessdata_fast) and
[`tessdata_best`](https://github.com/tesseract-ocr/tessdata_best) repositories;
the English file is the matching system-distributed Tesseract model. Tesseract
and its official data repositories are Apache-2.0 licensed.

No query images, checksums, or evaluator answers are embedded in these models.
