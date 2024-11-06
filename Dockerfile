FROM python:3.10.12
WORKDIR /app

COPY main.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["fastapi", "run", "--port", "8000"]
