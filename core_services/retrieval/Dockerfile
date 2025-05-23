FROM python:3.10-slim

WORKDIR /app

COPY core_services/retrieval/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY protos/ ./protos/
COPY core_services/retrieval/main.py .

EXPOSE 50055 9095

CMD ["python", "main.py"]