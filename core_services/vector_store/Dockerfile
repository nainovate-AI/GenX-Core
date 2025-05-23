FROM python:3.10-slim

WORKDIR /app

COPY core_services/vector_store/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY protos/ ./protos/
COPY core_services/vector_store/main.py .

EXPOSE 50054 9094

CMD ["python", "main.py"]