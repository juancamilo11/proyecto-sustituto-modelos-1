FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY train.py predict.py /app/

RUN pip install --no-cache-dir \
  numpy \
  pandas \
  scikit-learn \
  loguru

CMD ["python3", "train.py"]
