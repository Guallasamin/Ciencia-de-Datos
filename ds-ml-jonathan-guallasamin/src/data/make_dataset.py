"""
Script para descargar y extraer los datos originales del proyecto California Housing.

Descarga el archivo comprimido housing.tgz desde el repositorio de Aurélien Géron
y lo extrae en la carpeta data/raw/ para su uso posterior en el pipeline de datos.

Uso:
    python src/data/make_dataset.py
"""

import os
import urllib.request
import tarfile
from pathlib import Path


def fetch_housing_data(housing_url: str, housing_path: str):
    """
    Descarga y extrae los datos crudos del proyecto.

    Args:
        housing_url: URL del archivo .tgz con los datos.
        housing_path: Directorio donde se guardarán los datos extraídos.
    """
    # Crear el directorio de destino si no existe
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")

    # Descargar el archivo comprimido
    print(f"Descargando datos desde {housing_url} ...")
    urllib.request.urlretrieve(housing_url, tgz_path)

    # Extraer el contenido del .tgz
    print("Extrayendo datos...")
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)

    print("¡Descarga y extracción completadas con éxito!")


if __name__ == "__main__":
    URL = "https://github.com/ageron/data/raw/main/housing.tgz"
    PATH = "data/raw/"
    fetch_housing_data(URL, PATH)
