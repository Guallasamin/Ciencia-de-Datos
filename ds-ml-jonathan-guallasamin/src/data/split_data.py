"""
Script para dividir los datos en conjunto de entrenamiento, validación y prueba.

Metodología: Triple Split Estratificado (60% Train / 20% Validation / 20% Test)
basado en la variable 'median_income' para garantizar representatividad.

Esto evita data leakage y permite:
  - Train: entrenar modelos
  - Validation: comparar modelos y ajustar hiperparámetros
  - Test: evaluación final ÚNICA e imparcial
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

def split_and_save_data(raw_data_path: str, interim_data_path: str):
    """
    Realiza la partición estratificada de los datos en 3 conjuntos:
    Train (60%), Validation (20%), Test (20%).

    La estratificación se basa en categorías de ingreso medio (median_income)
    para asegurar que cada conjunto sea representativo de la distribución original.

    Args:
        raw_data_path: Ruta al archivo CSV con los datos crudos.
        interim_data_path: Ruta donde se guardarán los archivos particionados.
    """
    print(f"Leyendo datos desde {raw_data_path}...")
    housing = pd.read_csv(raw_data_path)
    print(f"Total de registros: {len(housing)}")
    
    # ─── Paso 1: Crear categorización de ingresos para muestreo estratificado ───
    # Se crean 5 categorías de ingreso para asegurar proporción en cada split
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )
    
    print("\nDistribución de categorías de ingreso en datos originales:")
    print(housing["income_cat"].value_counts(normalize=True).sort_index().round(4))
    
    # ─── Paso 2: Primer split - Separar Test Set (20%) ───────────────────────────
    # Primero separamos el 20% para Test, dejando 80% temporal
    split_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    for temp_index, test_index in split_test.split(housing, housing["income_cat"]):
        temp_set = housing.loc[temp_index]    # 80% temporal
        test_set = housing.loc[test_index]    # 20% test
    
    # ─── Paso 3: Segundo split - Separar Validation del Train (25% de 80% = 20% total) ─
    # Del 80% temporal, 25% va a Validation (0.25 × 0.80 = 0.20 del total)
    split_val = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    
    for train_index, val_index in split_val.split(temp_set, temp_set["income_cat"]):
        train_set = temp_set.loc[temp_set.index[train_index]]   # 60% del total
        val_set = temp_set.loc[temp_set.index[val_index]]       # 20% del total
    
    # ─── Paso 4: Eliminar la variable auxiliar income_cat ────────────────────────
    for set_ in (train_set, val_set, test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    
    # ─── Paso 5: Guardar los 3 conjuntos ─────────────────────────────────────────
    os.makedirs(interim_data_path, exist_ok=True)
    
    train_path = os.path.join(interim_data_path, "train_set.csv")
    val_path = os.path.join(interim_data_path, "val_set.csv")
    test_path = os.path.join(interim_data_path, "test_set.csv")
    
    train_set.to_csv(train_path, index=False)
    val_set.to_csv(val_path, index=False)
    test_set.to_csv(test_path, index=False)
    
    # ─── Paso 6: Reportar proporciones y verificar estratificación ───────────────
    total = len(train_set) + len(val_set) + len(test_set)
    print(f"\n{'='*60}")
    print(f"PARTICIÓN COMPLETADA")
    print(f"{'='*60}")
    print(f"  Train:      {len(train_set):>6} registros  ({len(train_set)/total*100:.1f}%)")
    print(f"  Validation: {len(val_set):>6} registros  ({len(val_set)/total*100:.1f}%)")
    print(f"  Test:       {len(test_set):>6} registros  ({len(test_set)/total*100:.1f}%)")
    print(f"  Total:      {total:>6} registros  (100.0%)")
    print(f"{'='*60}")
    
    # Verificar estratificación
    print("\nVerificación de estratificación (proporción de income_cat):")
    for name, df in [("Original", housing), ("Train", train_set), ("Validation", val_set), ("Test", test_set)]:
        # Recalcular income_cat para verificación
        cats = pd.cut(df["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
        proportions = cats.value_counts(normalize=True).sort_index()
        prop_str = "  ".join([f"Cat{i}: {p:.3f}" for i, p in proportions.items()])
        print(f"  {name:>10}: {prop_str}")
    
    print(f"\nArchivos guardados en: {interim_data_path}")
    print(f"  → {train_path}")
    print(f"  → {val_path}")
    print(f"  → {test_path}")

if __name__ == "__main__":
    RAW_PATH = "data/raw/housing/housing.csv"
    INTERIM_PATH = "data/interim/"
    split_and_save_data(RAW_PATH, INTERIM_PATH)
