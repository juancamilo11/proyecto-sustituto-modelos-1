import requests
import json

BASE_URL = 'http://localhost:5000'

print("\n1. Probando endpoint raíz (GET /)...")
response = requests.get(f"{BASE_URL}/")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")


print("\n2. Entrenando modelo (POST /train)...")

data = {
    "model_file": "model.pkl",
    "overwrite_model": "true",
    "max_depth": "3",
    "random_state": "42",
    "train_size": "0.8"
}

response = requests.post(
    f"{BASE_URL}/train",
    data=data
)

print(f"Status: {response.status_code}")
try:
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except:
    print("No se recibió JSON válido")

print("\n3. Haciendo predicciones del modelo (POST /predict)...")


