FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY train.py predict.py apirest.py /app/

RUN pip install --no-cache-dir \
  numpy \
  pandas \
  scikit-learn \
  loguru \
  flask

EXPOSE 5000

CMD ["python3", "apirest.py"]