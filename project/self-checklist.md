# Самопроверка проекта (self-checklist)

## Таблица самопроверки Smart Campus AirGuard

| # | Критерий | Статус | Где смотреть / комментарий |
| --- | --- | --- | --- |
| 1 | Сервис запускается по инструкции из `project/README.md` и работает | ✅ | `README.md`, запуск через `uvicorn`; Docker через `docker compose up --build`. |
| 2 | Endpoint `/predict` использует реальную модель, а не заглушку | ✅ | `src/airguard/service/app.py`, `src/airguard/models/inference.py`, `models/airguard_model.joblib`. |
| 3 | Есть EDA и хотя бы один эксперимент с метриками | ✅ | `notebooks/01_eda_and_experiments.ipynb`, `models/experiments.csv`, `report.md`. |
| 4 | Есть baseline и улучшенная модель, есть сравнение по метрикам | ✅ | `src/airguard/models/train.py`, `models/experiments.csv`, `report.md`. |
| 5 | Код не свален в один ноутбук, есть структура в `src/` | ✅ | `src/airguard/data`, `src/airguard/models`, `src/airguard/service`. |
| 6 | Есть Dockerfile или понятный сценарий развёртывания | ✅ | `Dockerfile`, `docker-compose.yml`, инструкции в `README.md`. |
| 7 | Есть `.env.example` и нет реальных секретов | ✅ | `configs/.env.example`, `SECURITY.md`. |
| 8 | Реализованы логи/наблюдаемость | ✅ | `logging` в `src/airguard/service/app.py`, `/health`, `/metrics`. |
| 9 | В `report.md` обоснован выбор финальной модели | ✅ | Разделы про эксперименты и выбор `gradient_boosting`. |
| 10 | `README.md` и `report.md` позволяют понять сценарий демонстрации | ✅ | Разделы запуска, API и демонстрации. |

Ориентировочная самооценка по чеклисту: 10/10 при условии, что зависимости установлены и команды из README проходят на машине проверяющего.
