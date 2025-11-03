import argparse
import pandas as pd
import numpy as np
import pickle
import os
from loguru import logger

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description='Entrenar modelo de clasificación para RENDIMIENTO_GLOBAL')

parser.add_argument('--data_file', required=True, type=str, help='archivo CSV con datos de entrenamiento')
parser.add_argument('--model_file', required=True, type=str, help='ruta donde se guardará el modelo entrenado')
parser.add_argument('--overwrite_model', default=False, action='store_true', help='sobrescribir el archivo del modelo si existe')
parser.add_argument('--max_depth', default=3, type=int, help='profundidad máxima del árbol de decisión')
parser.add_argument('--random_state', default=42, type=int, help='semilla aleatoria para reproducibilidad')
parser.add_argument('--train_size', default=0.8, type=float, help='proporción de datos para entrenamiento')

args = parser.parse_args()

if os.path.isfile(args.model_file):
    if args.overwrite_model:
        logger.info(f"Sobrescribiendo archivo de modelo existente: {args.model_file}")
    else:
        logger.error(f"El archivo del modelo {args.model_file} ya existe. Use --overwrite_model para sobrescribirlo")
        exit(-1)

logger.info(f"Cargando datos de entrenamiento desde {args.data_file}")
z = pd.read_csv(args.data_file)
logger.info(f"Shape del dataframe cargado: {z.shape}")

z = z[['ESTU_PRGM_DEPARTAMENTO', 'FAMI_ESTRATOVIVIENDA', 'RENDIMIENTO_GLOBAL']]

logger.info("Manejando valores faltantes")
z.FAMI_ESTRATOVIVIENDA.values[z.FAMI_ESTRATOVIVIENDA.isna()] = 'Sin Estrato'

logger.info("Aplicando mapeo de departamentos")
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
z.ESTU_PRGM_DEPARTAMENTO = z.ESTU_PRGM_DEPARTAMENTO.map(dept_map)

logger.info("Aplicando mapeo de estratos")
estrato_map = {
    "Estrato 1": 1, "Estrato 2": 2, "Estrato 3": 3,
    "Estrato 4": 4, "Estrato 5": 5, "Estrato 6": 6, "Sin Estrato": -1,
}
z.FAMI_ESTRATOVIVIENDA = z.FAMI_ESTRATOVIVIENDA.map(estrato_map)

logger.info("Aplicando one-hot encoding")
dept_vals = sorted(z.ESTU_PRGM_DEPARTAMENTO.unique())
dept_onehot = {val: np.eye(len(dept_vals))[i] for i, val in enumerate(dept_vals)}
dept_encoded = np.array([dept_onehot[i] for i in z.ESTU_PRGM_DEPARTAMENTO])
dept_df = pd.DataFrame(dept_encoded, columns=[f"ESTU_PRGM_DEPARTAMENTO__{v}" for v in dept_vals])

z = pd.concat([dept_df, z], axis=1).drop('ESTU_PRGM_DEPARTAMENTO', axis=1)

estrato_vals = sorted(z.FAMI_ESTRATOVIVIENDA.unique())
estrato_onehot = {val: np.eye(len(estrato_vals))[i] for i, val in enumerate(estrato_vals)}
estrato_encoded = np.array([estrato_onehot[i] for i in z.FAMI_ESTRATOVIVIENDA])
estrato_df = pd.DataFrame(estrato_encoded, columns=[f"FAMI_ESTRATOVIVIENDA__{v}" for v in estrato_vals])

z = pd.concat([estrato_df, z], axis=1).drop('FAMI_ESTRATOVIVIENDA', axis=1)

logger.info("Codificando variable objetivo")
y_col = 'RENDIMIENTO_GLOBAL'
rmap = {'alto': 3, 'bajo': 0, 'medio-bajo': 1, 'medio-alto': 2}
z[y_col] = z[y_col].map(rmap)

z = z[sorted(z.columns)]

X = z[[c for c in z.columns if c != y_col]].values
y = z[y_col].values
logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

logger.info(f"Dividiendo datos con train_size={args.train_size}")
Xtr, Xts, ytr, yts = train_test_split(X, y, train_size=args.train_size, random_state=args.random_state)
logger.info(f"Xtr: {Xtr.shape}, Xts: {Xts.shape}, ytr: {ytr.shape}, yts: {yts.shape}")

logger.info(f"Entrenando DecisionTreeClassifier con max_depth={args.max_depth}")
dt = DecisionTreeClassifier(max_depth=args.max_depth, random_state=args.random_state)
dt.fit(Xtr, ytr)
logger.info("Modelo entrenado exitosamente")

train_accuracy = accuracy_score(ytr, dt.predict(Xtr))
test_accuracy = accuracy_score(yts, dt.predict(Xts))

logger.info("="*60)
logger.info("RESULTADOS DEL MODELO")
logger.info("="*60)
logger.info(f"Accuracy en entrenamiento: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
logger.info(f"Accuracy en prueba: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
logger.info("="*60)

logger.info(f"Guardando modelo en {args.model_file}")
with open(args.model_file, 'wb') as f:
    pickle.dump(dt, f)
logger.success(f"Modelo guardado exitosamente en '{args.model_file}'")

logger.success("Proceso completado exitosamente")