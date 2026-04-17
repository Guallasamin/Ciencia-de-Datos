"""
Módulo para limpieza y enriquecimiento (Feature Engineering) usando Pipelines de Scikit-Learn.

Este módulo contiene:
    - drop_capped_values: Limpieza estricta de outliers/capping sobre el conjunto de entrenamiento.
    - CombinedAttributesAdder: Transformador para resolver multicolinealidad estructural (Ratios).
    - LogTransformer: Función paramétrica para abatir el Skewness asimétrico (>4).
    - get_preprocessing_pipeline(): Pipeline unificado para entrenamiento e inferencia.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def drop_capped_values(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Despliega la purga de censura artificial (capping) detectada en el EDA.
    
    Args:
        df: DataFrame original.
        is_train: Si es True, purga los registros para evitar sesgar los pesos del modelo.
                  Si es False (ej. Test/Val), se mantienen para reflejar el mundo real.
    
    Returns:
        DataFrame procesado.
    """
    df_clean = df.copy()
    if is_train:
        # Se remueven las observaciones con topes paramétricos exactos en la variable dependiente
        if 'median_house_value' in df_clean.columns:
            df_clean = df_clean[df_clean['median_house_value'] < 500001]
            
        # Topes de muestreo en edad
        if 'housing_median_age' in df_clean.columns:
            df_clean = df_clean[df_clean['housing_median_age'] < 52]
            
    return df_clean

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para disolver multicolinealidad espacial (r>0.85).
    """
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_df = isinstance(X, pd.DataFrame)
        
        if X_df:
            rooms = X['total_rooms'].values
            bedrooms = X['total_bedrooms'].values
            population = X['population'].values
            households = X['households'].values
        else:
            # Índices del array derivados del num_pipeline
            rooms = X[:, 3]
            bedrooms = X[:, 4]
            population = X[:, 5]
            households = X[:, 6]
            
        rooms_per_household = rooms / households
        population_per_household = population / households
        
        if self.add_bedrooms_per_room:
            bedrooms_per_room = bedrooms / rooms
            if X_df:
                X_ret = X.copy()
                X_ret["rooms_per_household"] = rooms_per_household
                X_ret["population_per_household"] = population_per_household
                X_ret["bedrooms_per_room"] = bedrooms_per_room
                return X_ret
            else:
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            if X_df:
                X_ret = X.copy()
                X_ret["rooms_per_household"] = rooms_per_household
                X_ret["population_per_household"] = population_per_household
                return X_ret
            else:
                return np.c_[X, rooms_per_household, population_per_household]


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Aplicación de logaritmo neperiano + 1 para reducción drástica de colas largas (skewness).
    """
    def __init__(self, cols_to_log):
        # Admite tanto nombres de columnas para DF como índices enteros para Numpy arrays
        self.cols_to_log = cols_to_log

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ret = X.copy()
        if isinstance(X_ret, pd.DataFrame):
            for col in self.cols_to_log:
                X_ret[col] = np.log1p(X_ret[col])
        else:
            for col in self.cols_to_log:
                X_ret[:, col] = np.log1p(X_ret[:, col])
        return X_ret


def get_preprocessing_pipeline() -> ColumnTransformer:
    """
    Pipeline unificado de extracción, enriquecimiento y compresión de simetrías.
    """
    # Índices basados en la entrada de num_attribs:
    # 0: longitude, 1: latitude, 2: age, 3: rooms, 4: bed, 5: pop, 6: househ, 7: income
    # Outputs de CombinedAttributesAdder agregan: 8: rooms_per_h, 9: pop_per_h, 10: bed_per_r
    # Variables a las que se aplicará logaritmo para simetrização (Skewness control):
    indices_skewed = [3, 4, 5, 6, 7, 8, 9] # Evitamos 'bedrooms_per_room' ratio cerrado < 1

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),               # Manejo de Nulos (MCAR)
        ('attribs_adder', CombinedAttributesAdder()),                # Transformación de Multicolinealidad
        ('log_transformer', LogTransformer(indices_skewed)),         # Simetrización de Skewness Extrema
        ('std_scaler', StandardScaler()),                            # Escalamiento Vectorial Geométrico
    ])
    
    cat_attribs = ["ocean_proximity"]
    num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms", 
                   "total_bedrooms", "population", "households", "median_income"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
    ])

    return full_pipeline


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Despliega un macro-resumen descriptivo conforme entra la data original.
    """
    print("=" * 60)
    print("REPORTE ANALÍTICO PRE-PIPELINE")
    print("=" * 60)
    
    # Valores nulos (MCAR demostrado)
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    if len(null_cols) > 0:
        print(f"\\n[!] Detección Faltantes (Mecanismo Estocástico):")
        for col, count in null_cols.items():
            print(f"    - {col}: {count} nulos ({count / len(df) * 100:.2f}%)")
    
    # Capping
    if "median_house_value" in df.columns:
        capped = (df["median_house_value"] >= 500001).sum()
        print(f"\\n[!] Censura Espacial Superior (Capping Target): {capped} registros ({capped / len(df) * 100:.2f}%)")
    
    # Duplicados
    n_dup = df.duplicated().sum()
    print(f"\\n[*] Exceso Duplicados Base: {n_dup}")
    print(f"\\n{'='*60}")
    
    return df.drop_duplicates()

if __name__ == "__main__":
    print("Módulo avanzado de Ingeniería de Features Algoritmizado cargado correctamente.")
    print("Fases Operativas: Capping Trimmer -> MCAR Median Imputation -> Ratio Extraction -> Log-Scalling -> Standarization -> OHE")
