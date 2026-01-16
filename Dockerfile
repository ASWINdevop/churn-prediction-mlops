FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .

RUN pip install --default-timeout=1000 --no-cache-dir --upgrade -r requirements.txt

COPY ./app ./app

COPY ./models ./models

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
