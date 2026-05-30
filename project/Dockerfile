FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV AIRGUARD_HOST=0.0.0.0
ENV AIRGUARD_PORT=8000
ENV PYTHONPATH=/app/project/src

WORKDIR /app/project

COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN python -m pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "pandas>=2.1.0" \
    "numpy>=1.26.0" \
    "scikit-learn>=1.4.0" \
    "joblib>=1.3.0" \
    "pydantic>=2.6.0" \
    "matplotlib>=3.8.0"

COPY configs ./configs
COPY data ./data
COPY models ./models
COPY README.md report.md self-checklist.md ./

RUN python -m airguard.data.generate_dataset --rows 6000 --output data/processed/airguard_dataset.csv \
    && python -m airguard.models.train --data data/processed/airguard_dataset.csv --model models/airguard_model.joblib

EXPOSE 8000

CMD ["uvicorn", "airguard.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
