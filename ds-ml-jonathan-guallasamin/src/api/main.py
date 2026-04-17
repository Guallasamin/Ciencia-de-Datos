"""
API Básica usando FastAPI para servir el modelo entrenado.

Endpoints:
  GET  /         → Mensaje de bienvenida
  GET  /health   → Estado de salud de la API y modelo
  GET  /model-info → Información sobre el modelo cargado
  POST /predict  → Predicción de precio de vivienda
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

# Inicializamos la app
app = FastAPI(
    title="API de Predicción de Precios de Vivienda (California)",
    description="API que utiliza un modelo RandomForest entrenado para predecir el precio medio de viviendas en distritos de California.",
    version="1.0"
)

# Esquema de datos esperado por la API
class HousingFeatures(BaseModel):
    """Variables de entrada necesarias para la predicción."""
    longitude: float = Field(..., description="Longitud geográfica del distrito", ge=-125, le=-114)
    latitude: float = Field(..., description="Latitud geográfica del distrito", ge=32, le=42)
    housing_median_age: float = Field(..., description="Edad mediana de las viviendas en el distrito", ge=0, le=52)
    total_rooms: float = Field(..., description="Total de habitaciones en el distrito", ge=0)
    total_bedrooms: float = Field(..., description="Total de recámaras en el distrito", ge=0)
    population: float = Field(..., description="Población total del distrito", ge=0)
    households: float = Field(..., description="Total de hogares en el distrito", ge=1)
    median_income: float = Field(..., description="Ingreso medio del distrito (en decenas de miles de USD)", ge=0)
    ocean_proximity: str = Field(..., description="Proximidad al océano: NEAR BAY, <1H OCEAN, INLAND, NEAR OCEAN, ISLAND")

    class Config:
        json_schema_extra = {
            "example": {
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41.0,
                "total_rooms": 880.0,
                "total_bedrooms": 129.0,
                "population": 322.0,
                "households": 126.0,
                "median_income": 8.3252,
                "ocean_proximity": "NEAR BAY"
            }
        }

# Variable global para cargar el modelo
model = None

@app.on_event("startup")
def load_model():
    """
    Carga el modelo globalmente al iniciar el servidor usando joblib.
    """
    global model
    # Ruta relativa al directorio raíz del proyecto (2 niveles arriba de src/api/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_path = os.path.join(project_root, "models", "best_model.pkl")
    try:
        model = joblib.load(model_path)
        print("✅ Modelo cargado exitosamente.")
    except Exception as e:
        print(f"⚠️ Advertencia: No se pudo cargar el modelo. Error: {e}")
        print("   ¿Ya lo entrenaste y guardaste con train_model.py?")

@app.get("/")
def home():
    """Endpoint raíz con mensaje de bienvenida."""
    return {
        "mensaje": "Bienvenido a la API del Proyecto Final de Ciencia de Datos",
        "endpoints": {
            "GET /health": "Estado de salud de la API",
            "GET /model-info": "Información sobre el modelo cargado",
            "POST /predict": "Predicción de precio de vivienda"
        }
    }

@app.get("/health")
def health_check():
    """Endpoint de salud para verificar que la API y el modelo están operativos."""
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "api_version": "1.0"
    }

@app.get("/model-info")
def model_info():
    """Retorna información del modelo cargado."""
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no se ha cargado.")
    
    info = {
        "tipo_pipeline": str(type(model).__name__),
        "pasos": [step[0] for step in model.steps] if hasattr(model, 'steps') else "N/A",
    }
    
    # Intentar obtener info del modelo final
    if hasattr(model, 'steps'):
        final_model = model.steps[-1][1]
        info["modelo_final"] = str(type(final_model).__name__)
        if hasattr(final_model, 'n_estimators'):
            info["n_estimators"] = final_model.n_estimators
        if hasattr(final_model, 'max_features'):
            info["max_features"] = final_model.max_features
    
    return info

@app.post("/predict")
def predict_price(features: HousingFeatures):
    """
    Recibe las características de un distrito y retorna la predicción de precio.
    
    El modelo espera las mismas columnas usadas durante el entrenamiento.
    El pipeline interno se encarga de la imputación, feature engineering,
    escalado y codificación categórica automáticamente.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no se ha cargado. Ejecuta train_model.py primero."
        )
    
    # Validar ocean_proximity
    valid_categories = ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"]
    if features.ocean_proximity not in valid_categories:
        raise HTTPException(
            status_code=400,
            detail=f"ocean_proximity inválido. Valores válidos: {valid_categories}"
        )
    
    # Convertir a DataFrame (el pipeline espera columnas nombradas)
    features_dict = features.model_dump()
    df_input = pd.DataFrame([features_dict])
    
    # Predecir
    prediction = model.predict(df_input)[0]
    
    return {
        "predicted_price": round(float(prediction), 2),
        "currency": "USD",
        "nota": "Predicción basada en datos del censo de California 1990"
    }

# Instrucciones para correr la API localmente:
# En la terminal, ejecuta:
# uvicorn src.api.main:app --reload
