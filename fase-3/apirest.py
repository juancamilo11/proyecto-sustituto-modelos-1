from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import pickle
import os
from loguru import logger
import subprocess

from train import train_model
from predict import predict_from_model

app = Flask(__name__)

MODEL_FILE = "model.pkl"

@app.route("/")
def hello_world():
  return jsonify({
      "message": "API REST para modelo ML",
      "endpoints": ["/predict", "/train"]
  })

@app.route("/train-model", methods=["POST"])
def train():
  try:
      data = request.get_json()

      if not data:
          return jsonify({"error": "El body debe ser JSON"}), 400

      model_file = data.get("model_file")
      if not model_file:
          return jsonify({"error": "Falta 'model_file'"}), 400

      overwrite_model = bool(data.get("overwrite_model", False))
      max_depth = int(data.get("max_depth", 3))
      random_state = int(data.get("random_state", 42))
      train_size = float(data.get("train_size", 0.8))

      result = train_model(
          model_file=model_file,
          overwrite_model=overwrite_model,
          max_depth=max_depth,
          random_state=random_state,
          train_size=train_size
      )

      return jsonify({
          "message": "Modelo entrenado y guardado correctamente",
          **result
      })

  except Exception as e:
      logger.exception("Error en /train")
      return jsonify({"error": str(e)}), 500


@app.route("/predict-model", methods=["POST"])
def predict():
  try:
    data = request.get_json()

    if not data:
        return jsonify({"error": "El body debe ser JSON"}), 400

    model_file = data.get("model_file")
    if not model_file:
        return jsonify({"error": "Falta 'model_file'"}), 400

    result = predict_from_model(
        model_file=model_file,
    )

    return jsonify({
        "message": "Predicción realizada correctamente",
        **result
    })

  except Exception as e:
      logger.exception("Error en /predict")
      return jsonify({"error": str(e)}), 500