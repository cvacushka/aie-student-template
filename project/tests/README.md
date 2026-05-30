# Тесты Smart Campus AirGuard

В проект добавлены pytest-тесты:

- `test_data_generation.py` — проверяет форму синтетического датасета, наличие целевого признака и валидацию;
- `test_training.py` — проверяет, что обучение создаёт реальный model artifact и метрики;
- `test_api.py` — проверяет, что `/predict` работает с загруженной моделью.

Запуск:

```powershell
cd project
python -m pytest
```
