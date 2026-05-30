# Код Smart Campus AirGuard

Основной код проекта находится в пакете `src/airguard/`:

- `data/generate_dataset.py` — генерация синтетического датасета;
- `data/validate.py` — проверки качества данных;
- `features.py` — построение признаков;
- `models/train.py` — обучение и сравнение моделей;
- `models/inference.py` — загрузка артефакта и прогноз;
- `service/app.py` — FastAPI-приложение с `/health`, `/predict`, `/metrics`;
- `service/metrics.py` — простые Prometheus-like метрики;
- `schemas.py` — Pydantic-схемы API.

Основные команды:

```powershell
cd project
python -m airguard.data.generate_dataset --rows 6000
python -m airguard.models.train
python -m uvicorn airguard.service.app:app --reload
```
