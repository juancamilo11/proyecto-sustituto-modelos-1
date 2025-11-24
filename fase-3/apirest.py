from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import pickle
import os
from loguru import logger
import subprocess

from train import train_model

app = Flask(__name__)

MODEL_FILE = "model.pkl"

@app.route("/")
def hello_world():
  return jsonify({
      "message": "API REST para modelo ML",
      "endpoints": ["/predict", "/train"]
  })

@app.route("/train", methods=["POST"])
def train():
    try:
        model_file = request.form.get("model_file")
        if not model_file:
            return jsonify({"error": "Falta 'model_file'"}), 400

        overwrite_model = request.form.get("overwrite_model", "false").lower() == "true"
        max_depth = int(request.form.get("max_depth", 3))
        random_state = int(request.form.get("random_state", 42))
        train_size = float(request.form.get("train_size", 0.8))

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




@app.route("/predict", methods=['POST'])
def predict():
    try:
      if not os.path.isfile(MODEL_FILE):
        return jsonify({'error': f'Modelo {MODEL_FILE} no existe. Ejecute /train primero'}), 400

      
          
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)