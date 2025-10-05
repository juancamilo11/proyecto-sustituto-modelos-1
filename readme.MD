
Proyecto Sustituto | Modelos 1 | Entrega 1
==============================

**Proyecto del Curso:** Predicción de Rendimiento Académico con Machine Learning
--------------------------------------------------------

Este proyecto implementa un modelo de Machine Learning para predecir el rendimiento académico global de estudiantes basándose en características demográficas y socioeconómicas.


Descripción del Proyecto
---------------------------

El modelo predice el `RENDIMIENTO_GLOBAL` de estudiantes en 4 categorías:

*   **Bajo** (0)
*   **Medio-bajo** (1)
*   **Medio-alto** (2)
*   **Alto** (3)

### Variables de entrada:

*   `ESTU_PRGM_DEPARTAMENTO`: Departamento del programa académico
*   `FAMI_ESTRATOVIVIENDA`: Estrato socioeconómico de la vivienda


Ejecución en Google Colab
----------------------------

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

### Listo, ya se tiene todo el ambiente listo para el entrenamiendo y las predicciones.

Desarrollo Futuro: API REST
------------------------------

Este proyecto será adaptado para funcionar como un servicio API REST que permitirá:

### Endpoints planificados:

1.  **POST /api/train**
    *   Entrenar el modelo con nuevos datos
    *   Parámetros de configuración del modelo
2.  **POST /api/test**
    *   Realizar predicciones individuales o por lotes
    *   Input: características del estudiante
    *   Output: predicción de rendimiento

### Tecnologías a utilizar:

*   **FastAPI** o **Flask** para el backend
*   **Docker** para containerización
*   **PostgreSQL** (En caso de ser necesario) para persistencia de datos

### Arquitectura propuesta:

    Cliente (Web/Mobile)
        ↓
        Proxy reverso de NGINX
        ↓
    FastAPI Backend
        ↓
    ├── Servicio de Entrenamiento
    ├── Servicio de Predicción
    └── Base de Datos (En caso de ser necesario)

-----------------

Integrantes
-----------------
Cardona Calderón Maria Carolina

Cardona Calderón Juan Camilo

**Curso**: Modelos 1 - Facultad de Ingeniería - Universidad de Antioquia