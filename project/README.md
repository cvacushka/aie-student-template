# Smart Campus AirGuard

Итоговый проект по курсу «Инженерия Искусственного Интеллекта».

## Паспорт

- **Название проекта:** Smart Campus AirGuard
- **Автор:** Шатохина Дарья Евгеньевна
- **Группа:** ИКБО-13-23
- **Контакт:** @Daryushka_Shat
- **Тип задачи:** бинарная классификация риска ухудшения качества воздуха в аудитории.
- **Пользователь:** диспетчер учебного корпуса или администратор smart campus.

Smart Campus AirGuard — учебный end-to-end AI-сервис для прогноза риска ухудшения качества воздуха в аудитории. Сервис получает текущие показания аудитории, расписания и вентиляции, а возвращает вероятность того, что через 30 минут CO2/дискомфорт выйдет в рискованную зону. Данные синтетические и обезличенные, модели обучаются локально, результат доступен через FastAPI.

## Структура проекта

```text
project/
  configs/              # конфиги и .env.example
  data/                 # синтетические учебные данные и описание данных
  examples/             # готовые JSON-примеры для API
  models/               # локальные артефакты модели и метрики
  notebooks/            # EDA и эксперименты
  src/airguard/         # генерация данных, обучение, inference и API
  tests/                # pytest smoke/unit tests
  Dockerfile
  docker-compose.yml
  pyproject.toml
  requirements.txt
  README.md
  report.md
  self-checklist.md
  SECURITY.md
```

Дополнительная папка `models/` используется для сериализованной модели `airguard_model.joblib`, таблицы экспериментов `experiments.csv` и итоговых метрик `metrics.json`.

## Быстрый запуск локально

Нужен Python 3.10 или новее.

```powershell
cd project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m airguard.data.generate_dataset --rows 6000
python -m airguard.models.train
python -m uvicorn airguard.service.app:app --reload
```

После запуска:

- Swagger UI: <http://127.0.0.1:8000/docs>
- Health-check: <http://127.0.0.1:8000/health>
- Метрики: <http://127.0.0.1:8000/metrics>

Быстрая проверка работоспособности:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/metrics"
```

## Быстрая удалённая проверка

Если проект проверяется без участия автора, достаточно выполнить команды ниже из свежего клона репозитория:

```bash
cd project
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
python -m uvicorn airguard.service.app:app --host 127.0.0.1 --port 8000
```

В другом терминале можно проверить API:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data @examples/predict_payload.json
curl http://127.0.0.1:8000/metrics
```

Ожидаемо `/health` возвращает `status: ok`, `/predict` возвращает вероятность риска и рекомендацию, а `/metrics` показывает счётчики запросов.

## Пример запроса к `/predict`

```powershell
$body = @{
  timestamp = "2026-03-17T10:35:00"
  room_id = "B-421"
  room_type = "lecture"
  building_zone = "central"
  lesson_type = "lecture"
  room_area_m2 = 72.0
  capacity = 70
  occupancy_count = 61
  hvac_airflow_m3h = 360.0
  ventilation_level = 0.38
  outdoor_temp_c = 6.0
  indoor_temp_c = 25.8
  humidity_pct = 58.0
  co2_ppm_current = 1040.0
  minutes_since_lesson_start = 48
  minutes_until_break = 32
  noise_level_db = 59.0
  is_exam_period = $false
  floor = 4
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## Docker

```powershell
cd project
docker compose up --build
```

Dockerfile генерирует синтетический датасет и обучает модель при сборке образа.

## Тесты

```powershell
cd project
python -m pytest
```

Тесты проверяют генерацию данных, обучение реальной модели и рабочий `/predict`.

## Контрольная проверка перед сдачей

```powershell
cd project
python -m pip install -r requirements.txt
python -m airguard.data.generate_dataset --rows 6000
python -m airguard.models.train
python -m pytest
python -m uvicorn airguard.service.app:app --reload
```

Ожидаемый результат: тесты проходят, `/health` возвращает `status: ok`, `/predict` отдаёт вероятность риска, а `/metrics` показывает счётчики HTTP-запросов и предсказаний.

## Демонстрационный сценарий защиты

1. Показать постановку задачи и ограничения в `README.md` и `report.md`.
2. Сгенерировать данные и обучить модель.
3. Открыть `/health` и убедиться, что модель загружена.
4. Отправить пример в `/predict` через Swagger UI.
5. Открыть `/metrics` и показать, что счётчики запросов обновляются.

## Ограничения и развитие

- Данные синтетические, поэтому качество модели не доказывает готовность к промышленной эксплуатации.
- Нет онлайн-мониторинга drift и автоматического переобучения.
- Нет интеграции с реальной BMS/HVAC-системой кампуса.
- Дальше можно подключить реальные обезличенные сенсорные данные, добавить batch endpoint и трекинг экспериментов в MLflow.
