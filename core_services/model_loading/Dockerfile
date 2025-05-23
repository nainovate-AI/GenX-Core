FROM python:3.10-slim

WORKDIR /app

COPY core_services/model_loading/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY protos/ ./protos/
COPY core_services/model_loading/main.py .

EXPOSE 50053 9093

CMD ["python", "main.py"]