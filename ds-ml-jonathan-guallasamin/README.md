# Proyecto Final de Fundamentos de Ciencia de Datos - USFQ
**Predicción de Precios de Viviendas en California**

Este repositorio contiene la solución completa para predecir los precios medios de viviendas en distritos de California. Se implementó un pipeline end-to-end desde la recolección y preparación de datos hasta el despliegue de un modelo `RandomForestRegressor` mediante un microservicio en `FastAPI`.

---

## Memoria Técnica de Ejecución

### Fase 1: Análisis Exploratorio y Calidad de Datos (15 pts)

1. **Recolección y Partición de Datos:**
   - Se implementó `src/data/make_dataset.py` para descargar el conjunto housing.tgz.
   - Se ejecutó `src/data/split_data.py` aplicando partición estratificada según la variable de ingresos medios. El conjunto se dividió en Train/Validation/Test, asegurando que los conjuntos de validación y prueba permanecieran excluidos durante el entrenamiento para mitigar fuga de datos.

2. **Análisis Exploratorio (`01_exploracion.ipynb`):**
   - Se realizó exploración descriptiva sobre el conjunto de entrenamiento.
   - Se identificaron tres deficiencias estructurales:
     - Valores faltantes MCAR (Missing Completely At Random) en la característica total_bedrooms.
     - Valores truncados en precio máximo ($500,001) y antigüedad máxima (52 años).
     - Asimetría en distribuciones de variables y multicolinealidad territorial.

### Fase 2: Limpieza y Feature Engineering (15 pts)

1. **Procesamiento de Datos (`02_limpieza_enriquecimiento.ipynb`):**
   - Los valores faltantes se imputaron usando la mediana del conjunto de entrenamiento.
   - Se crearon características derivadas para reducir multicolinealidad: `rooms_per_household`, `population_per_household`, `bedrooms_per_room`.
   - Se aplicó transformación logarítmica log(1+x) para estabilizar distribuciones asimétricas.

2. **Pipeline de Producción (`src/features/build_features.py`):**
   - Se implementó un pipeline de Scikit-Learn que encapsula todas las transformaciones.
   - Se desarrolló la función `drop_capped_values` que remueve observaciones con valores truncados durante el entrenamiento, sin aplicarse al conjunto de prueba para preservar la evaluación en datos no modificados.

### Fase 3: Experimentación y Selección de Modelos (25 pts)

1. **Comparación de Algoritmos (`03_experimentacion.ipynb`):**
   - Se entrenaron cuatro algoritmos: LinearRegression, SGDRegressor, DecisionTreeRegressor y RandomForestRegressor.
   - DecisionTreeRegressor exhibió sobreajuste severo (RMSE=0 en entrenamiento). Se aplicaron restricciones paramétricas (`max_depth=12`, `min_samples_leaf=10`) para regularizar el modelo.
   - Se realizó validación cruzada con K=5 exclusivamente en el conjunto de entrenamiento.

2. **Optimización de Hiperparámetros:**
   - Se seleccionó RandomForestRegressor como arquitectura ganadora y se aplicó GridSearchCV para optimización.
   - Los hiperparámetros óptimos fueron `bootstrap=False` y `n_estimators=200`.
   - En el conjunto de prueba, el modelo alcanzó un error promedio (MAE) de aproximadamente $53,273 USD con R² consistente.

3. **Serialización del Modelo (`train_model.py`):**
   - Se implementó el script de entrenamiento que carga los conjuntos de train y validación, aplica el filtro de valores truncados, ejecuta el pipeline y serializa el modelo mediante joblib en formato `.pkl`.

### Fase 4: Despliegue con FastAPI (15 pts)

Se desarrolló un microservicio en `src/api/main.py` para servir predicciones. El servicio valida entradas utilizando esquemas Pydantic y aplica el pipeline de características antes de la predicción.

#### Iniciación del Servidor
```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```
El servidor se ejecuta en `http://localhost:8000`.

#### Interfaz Swagger UI
Se accede a la documentación interactiva en `http://localhost:8000/docs` donde se pueden probar los endpoints: `/`, `/health`, `/model-info` y `/predict`.

#### Predicción mediante cURL
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41,
  "total_rooms": 880,
  "total_bedrooms": 129,
  "population": 322,
  "households": 126,
  "median_income": 8.3252,
  "ocean_proximity": "NEAR BAY"
}'
```

Respuesta:
```json
{"predicted_price": 452600.0, "currency": "USD"}
```

#### Cliente Python
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "longitude": -118.25,
    "latitude": 34.05,
    "housing_median_age": 20,
    "total_rooms": 2500,
    "total_bedrooms": 500,
    "population": 1200,
    "households": 450,
    "median_income": 4.5,
    "ocean_proximity": "<1H OCEAN"
}

response = requests.post(url, json=data)
if response.status_code == 200:
    pred = response.json()
    print(f"Precio estimado: ${pred['predicted_price']} {pred['currency']}")
```

#### Flujo de Validación y Predicción
1. FastAPI valida el esquema JSON contra las restricciones Pydantic. Valores inválidos devuelven HTTP 400.
2. El diccionario validado se convierte a DataFrame y se procesa mediante el pipeline desserializado.
3. El modelo genera la predicción aplicando las transformaciones capturadas durante el entrenamiento.
4. La respuesta se retorna en formato JSON.

---