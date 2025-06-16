FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y curl gnupg bash git && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://ollama.com/install.sh | sh
ENV PATH="/root/.ollama/bin:${PATH}"

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["sh", "-c", "ollama serve & until ollama run phi3 < /dev/null; do echo 'Waiting for model ...'; sleep 1; done && uvicorn app:app --host 0.0.0.0 --port 8000"]
