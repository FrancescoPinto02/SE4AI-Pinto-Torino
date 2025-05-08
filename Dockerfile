# Dockerfile
FROM python:3.10-slim

# Installa build tools per compilare scikit-surprise
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

# Crea working directory
WORKDIR /app

# Copia requirements e installa dipendenze
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia il codice e i dati
COPY src/ ./src/
COPY .env .env
COPY data/processed/ data/processed/
COPY data/raw/ data/raw/

EXPOSE 8000

# Avvio FastAPI
CMD ["uvicorn", "src.recommender.api:app", "--host", "0.0.0.0", "--port", "8000"]