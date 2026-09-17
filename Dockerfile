# Image légère avec Python. Le modèle est exécuté sur CPU sur Render.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Cette étape reste en cache tant que les dépendances ne changent pas.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# Render fournit la variable PORT (10000 par défaut).
EXPOSE 10000
CMD ["sh", "-c", "uvicorn API.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
