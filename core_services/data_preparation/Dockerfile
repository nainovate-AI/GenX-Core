FROM python:3.10-slim

WORKDIR /app

COPY core_services/data_preparation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core_services/data_preparation/main.py .

COPY protos/ ./protos/

EXPOSE 50051
CMD ["python", "main.py"]
