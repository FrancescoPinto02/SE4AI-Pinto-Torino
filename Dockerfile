FROM python:3.10-slim

# Installa build tools per compilare scikit-surprise
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia requirements e installa dipendenze
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Installazione DVC con supporto S3
RUN pip install --no-cache-dir dvc[s3]

# Copia i file di configurazione DVC e il codice
COPY .dvc/ .dvc/
COPY dvc.yaml dvc.yaml
COPY dvc.lock dvc.lock
COPY src/ ./src/

# Imposta variabili d'ambiente per S3 (legate a GitHub Secrets in CD)
ENV AWS_ACCESS_KEY_ID=${DVC_S3_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${DVC_S3_SECRET_ACCESS_KEY}

# Pull dei dati dal remote
RUN dvc pull -r origin

EXPOSE 8000

CMD ["uvicorn", "src.recommender.api:app", "--host", "0.0.0.0", "--port", "8000"]
