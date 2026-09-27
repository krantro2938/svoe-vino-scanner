# Решения и открытые вопросы

## Принятые решения

| Решение | Причина |
| --- | --- |
| CPU-first local path обязателен | На текущей машине NVIDIA driver недоступен; приватный прогон должен работать без внешних ресурсов |
| Exact search вместо ANN в hot path | 2 103 позиции достаточно малы; exact search детерминирован и не снижает recall |
| Rich product API отделён от evaluator API | UI нужны confidence/status/card, официальный скрипт принимает flat slug |
| Unresolved/ambiguous media не индексируются автоматически | Ошибочная reference label отравляет метрики сильнее, чем временно неполный index |
| Sommelier deterministic-first | Retention feature должна работать без сети и не выдумывать wine facts |
| Browser storage только для history | Сервер не должен сохранять пользовательские фотографии по умолчанию |

## Требуют ответа кейсодержателя

1. Все ли private images гарантированно имеют slug в переданном CSV snapshot?
2. Какой ответ ожидается для out-of-catalog query: `null`, analogue или current live-catalog slug?
3. Какой slug canonical, если одна reference photo относится к нескольким slug?
4. Что именно называется «Top-1 и Top-5 F1»: aggregate F1/Recall@5 или candidate confidence?
5. Разрешён ли offline snapshot дополнительных полей public portal cards?
6. Какое железо и наличие сети гарантируются на private online run?

До ответа product endpoint использует honest uncertain/not-found semantics, а evaluator endpoint всегда выдаёт лучший in-catalog slug.

