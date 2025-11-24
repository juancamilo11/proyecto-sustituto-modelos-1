import argparse
import numpy as np
import pandas as pd
import pickle
import os
from loguru import logger


def predict_from_model(
  model_file
):
  if not os.path.isfile(model_file):
      raise FileExistsError(f"El archivo del modelo {model_file} no existe")

  if not os.path.isfile("test.csv"):
      raise FileExistsError(f"El archivo de entrada test.csv no existe")

  logger.info(f"Cargando datos de entrada desde test.csv")
  zt = pd.read_csv("test.csv")
  logger.info(f"Shape de datos cargados: {zt.shape}")

  if 'ID' in zt.columns:
      zt_ids = zt.ID.values
      logger.info(f"IDs encontrados: {len(zt_ids)}")
  else:
      zt_ids = np.arange(len(zt))
      logger.warning("No se encontró columna 'ID', usando índices secuenciales")

  zt = zt[['ESTU_PRGM_DEPARTAMENTO', 'FAMI_ESTRATOVIVIENDA']]

  logger.info("Aplicando transformaciones a datos de entrada")

  zt.FAMI_ESTRATOVIVIENDA.values[zt.FAMI_ESTRATOVIVIENDA.isna()] = 'Sin Estrato'

  dept_map = {
      "BOGOTÁ": "BOG", "ANTIOQUIA": "ANT", "VALLE": "VAL", "ATLANTICO": "ATL",
      "SANTANDER": "SAN", "NORTE SANTANDER": "NSA", "BOLIVAR": "BOL", "BOYACA": "BOY",
      "CUNDINAMARCA": "CUN", "NARIÑO": "NAR", "RISARALDA": "RIS", "CORDOBA": "COR",
      "TOLIMA": "TOL", "CALDAS": "CAL", "CAUCA": "CAU", "HUILA": "HUI",
      "MAGDALENA": "MAG", "SUCRE": "SUC", "CESAR": "CES", "QUINDIO": "QUI",
      "META": "MET", "LA GUAJIRA": "GUA", "CHOCO": "CHO", "CAQUETA": "CAQ",
      "CASANARE": "CAS", "PUTUMAYO": "PUT", "ARAUCA": "ARA", "AMAZONAS": "AMA",
      "GUAVIARE": "GUV", "VAUPES": "VAU", "SAN ANDRES": "SAD",
  }
  zt.ESTU_PRGM_DEPARTAMENTO = zt.ESTU_PRGM_DEPARTAMENTO.map(dept_map)

  estrato_map = {
      "Estrato 1": 1, "Estrato 2": 2, "Estrato 3": 3,
      "Estrato 4": 4, "Estrato 5": 5, "Estrato 6": 6, "Sin Estrato": -1,
  }
  zt.FAMI_ESTRATOVIVIENDA = zt.FAMI_ESTRATOVIVIENDA.map(estrato_map)

  dept_vals = sorted(zt.ESTU_PRGM_DEPARTAMENTO.unique())
  dept_onehot = {val: np.eye(len(dept_vals))[i] for i, val in enumerate(dept_vals)}
  dept_encoded = np.array([dept_onehot[i] for i in zt.ESTU_PRGM_DEPARTAMENTO])
  dept_df = pd.DataFrame(dept_encoded, columns=[f"ESTU_PRGM_DEPARTAMENTO__{v}" for v in dept_vals])

  zt = pd.concat([dept_df, zt], axis=1).drop('ESTU_PRGM_DEPARTAMENTO', axis=1)

  estrato_vals = sorted(zt.FAMI_ESTRATOVIVIENDA.unique())
  estrato_onehot = {val: np.eye(len(estrato_vals))[i] for i, val in enumerate(estrato_vals)}
  estrato_encoded = np.array([estrato_onehot[i] for i in zt.FAMI_ESTRATOVIVIENDA])
  estrato_df = pd.DataFrame(estrato_encoded, columns=[f"FAMI_ESTRATOVIVIENDA__{v}" for v in estrato_vals])

  zt = pd.concat([estrato_df, zt], axis=1).drop('FAMI_ESTRATOVIVIENDA', axis=1)

  X_input = zt[sorted(zt.columns)].values
  logger.info(f"Shape de datos procesados: {X_input.shape}")

  logger.info(f"Cargando modelo desde {model_file}")
  with open(model_file, 'rb') as f:
      model = pickle.load(f)
  logger.info("Modelo cargado exitosamente")

  logger.info("Generando predicciones")
  preds = model.predict(X_input)
  logger.info(f"Predicciones generadas: {len(preds)}")

  rmap_inverse = {0: 'bajo', 1: 'medio-bajo', 2: 'medio-alto', 3: 'alto'}
  text_preds = [rmap_inverse[i] for i in preds]

  predictions_df = pd.DataFrame({
      'ID': zt_ids,
      'RENDIMIENTO_GLOBAL': text_preds
  })

  summary = predictions_df["RENDIMIENTO_GLOBAL"].value_counts().to_dict()

  return {
      "total": len(predictions_df),
      "summary": summary,
      "predictions": predictions_df.to_dict(orient="records")
  }
