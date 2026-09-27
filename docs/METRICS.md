# Метрики

## Определения

- **Accuracy@1**: доля query, где первый slug точно равен ground truth.
- **Micro-F1**: для single-label multiclass задачи равен Accuracy@1.
- **Macro-F1**: среднее F1 по классам; показывается только при достаточном labelled coverage.
- **Recall@5**: доля query, где ground truth присутствует среди первых пяти slug.
- **MRR@5**: среднее обратного ранга ground truth в Top-5.
- **Margin**: score Top-1 минус score Top-2. Это не F1.
- **Latency**: p50, p95 и max полного последовательного запроса; cold start публикуется отдельно.

## Текущий статус

Public manifest содержит три query без ground truth. Поэтому текущий публичный прогон может проверить schema, checksum, slug validity, completeness и latency, но не F1/Accuracy.

Заполнить после создания labelled field set:

| Version | Catalog checksum | Queries | Accuracy@1 | Macro-F1 | Recall@5 | p50 | p95 | Hardware |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pending | pending | pending | — | — | — | — | — | CPU-first demo machine |

## Публичный integrity-прогон

Неизменённый `participant_test.sh` успешно обработал 3/3 query. Получены три
валидных catalog slug. После staged-OCR оптимизации локальный cold CPU-прогон занял
1876, 2071 и 2226 мс (p50 2071 мс, max 2226 мс). Полный восьмирегионный OCR ранее
имел median 3884 мс; Top-1 slug на всех трёх фото сохранился. Confidence теперь
0.69, 0.75 и 0.69 вместо ложного насыщения на 0.99. Это проверка
completeness/latency и честности confidence, а не Accuracy: `expected_slug` в
public manifest отсутствует.

Для labelled проверки добавлен воспроизводимый synthetic workflow с неизменяемым
group-level split, отдельными split metrics и hard-negative slices. Он не заменяет
финальный физический field-photo holdout.

Каждый опубликованный результат должен ссылаться на immutable query manifest, model/index/catalog versions, preprocessing configuration и exact command.
