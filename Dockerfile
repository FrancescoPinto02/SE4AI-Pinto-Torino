FROM python:3.10-slim

# Installa build tools necessari per scikit-surprise
RUN apt-get update && apt-get install -y \
    build-essential gcc g++ libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

# Crea la working dir
WORKDIR /app

# Copia requirements e installa dipendenze
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia codice sorgente
COPY src/ ./src/

# Esponi la porta FastAPI
EXPOSE 8000

# Avvia l'API
CMD ["uvicorn", "src.recommender.api:app", "--host", "0.0.0.0", "--port", "8000"]
