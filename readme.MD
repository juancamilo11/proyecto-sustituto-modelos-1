# Proyecto Sustituto | Modelos 1 | Entrega 3

**Proyecto del Curso:** Predicción de Rendimiento Académico con Machine Learning
--------------------------------------------------------

Este proyecto implementa un modelo de Machine Learning para predecir el rendimiento académico global de estudiantes basándose en características demográficas y socioeconómicas.


## Descripción del Proyecto

El modelo predice el `RENDIMIENTO_GLOBAL` de estudiantes en 4 categorías:

*   **Bajo** (0)
*   **Medio-bajo** (1)
*   **Medio-alto** (2)
*   **Alto** (3)

### Variables de entrada:

*   `ESTU_PRGM_DEPARTAMENTO`: Departamento del programa académico
*   `FAMI_ESTRATOVIVIENDA`: Estrato socioeconómico de la vivienda


## Ejecución en Google Colab

### Paso 1: Abrir Google Colab

1.  Ir a [Google Colab](https://colab.research.google.com/)
2.  Inicia sesión con tu cuenta de Google
3.  Crea un nuevo notebook: **Archivo → Nuevo notebook**

### Paso 2: Instalar Dependencias

Ejecuta la siguiente celda para instalar las librerías necesarias:

```bash
!pip install kaggle rlxutils -q
```

### Paso 3: Configurar Credenciales de Kaggle

#### 3.1 Obtener el archivo `kaggle.json`

1.  Ir a [Kaggle](https://www.kaggle.com/)
2.  Inicia sesión en tu cuenta
3.  Haz clic en tu **foto de perfil** (esquina superior derecha)
4.  Selecciona **Settings** (Configuración)
5.  En la sección **API**, haz clic en **Create New Token**
6.  Se descargará automáticamente el archivo `kaggle.json`

#### 3.2 Subir el archivo a Colab

1.  En el panel izquierdo de Colab, haz clic en el ícono de **carpeta**
2.  Haz clic en el botón **Upload** (subir)
3.  Selecciona el archivo `kaggle.json` que descargaste

### Listo, ya se tiene todo el ambiente listo para el entrenamiento y las predicciones.

---------------------


## Ejecución con Docker

### Scripts:

*   **train.py**: Entrena el modelo DecisionTreeClassifier con los datos de `train.csv` y guarda el modelo en `model.pkl`
*   **predict.py**: Carga el modelo entrenado y genera predicciones sobre `test.csv`

### Uso:

```bash
# Construir imagen
docker-compose build

# Entrenar modelo
docker-compose run --rm train

# Generar predicciones
docker-compose run --rm predict
```

### Archivos Docker:

**Dockerfile**: Imagen Python 3.10 con dependencias (numpy, pandas, scikit-learn, loguru, flask)

**docker-compose.yml**: Define servicios para la API REST


---------------------


## API REST (Entrega 3)

El proyecto ahora incluye una API REST construida con Flask que permite entrenar el modelo y realizar predicciones mediante endpoints HTTP.

### Iniciar la API:

```bash
# Levantar el servicio API
docker-compose up api --build

# La API estará disponible en http://localhost:5000
```

### Endpoints disponibles:

#### 1. **GET /**
Información básica de la API

```bash
curl http://localhost:5000/
```

#### 2. **POST /train**
Entrena el modelo con los datos de `train.csv`

**Parámetros:**
- `model_file`: Nombre del archivo del modelo (requerido)
- `overwrite_model`: Sobrescribir modelo existente (true/false, default: false)
- `max_depth`: Profundidad máxima del árbol (default: 3)
- `random_state`: Semilla aleatoria (default: 42)
- `train_size`: Proporción de datos de entrenamiento (default: 0.8)

**Ejemplo:**

```bash
curl -X POST http://localhost:5000/train \
  -F "model_file=model.pkl" \
  -F "overwrite_model=true" \
  -F "max_depth=3" \
  -F "random_state=42" \
  -F "train_size=0.8"
```

#### 3. **POST /predict**
Genera predicciones sobre los datos de `test.csv`

**Parámetros:**
- `model_file`: Nombre del archivo del modelo (requerido)

**Ejemplo:**

```bash
curl -X POST http://localhost:5000/predict \
  -F "model_file=model.pkl"
```

**Respuesta:** JSON con el total de predicciones, resumen por categoría y lista completa de predicciones con IDs.

### Cliente de prueba:

Se incluye el script `client.py` para probar todos los endpoints:

```bash
python client.py
```

Este script realiza las siguientes operaciones:
1. Consulta el endpoint raíz
2. Entrena el modelo
3. Genera predicciones

---------------------


## Estructura del Proyecto

```
.
├── train.py              # Script de entrenamiento
├── predict.py            # Script de predicción
├── apirest.py            # API REST con Flask
├── client.py             # Cliente de prueba para la API
├── Dockerfile            # Imagen Docker
├── docker-compose.yaml   # Configuración de servicios
├── train.csv             # Datos de entrenamiento
├── test.csv              # Datos de prueba
└── model.pkl             # Modelo entrenado (generado)
```

---------------------


## Integrantes

Cardona Calderón Maria Carolina

Cardona Calderón Juan Camilo

**Curso**: Modelos 1 - Facultad de Ingeniería - Universidad de Antioquia