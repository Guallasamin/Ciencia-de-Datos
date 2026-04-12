# Fundamentos de Ciencia de Datos - Semana 2: EDA & Data Wrangling

Este repositorio contiene el material principal para la clase de **Análisis Exploratorio de Datos (EDA)** y **Data Wrangling** impartida en la maestría dentro de la asignatura de **Fundamentos de Ciencia de Datos** (Universidad San Francisco de Quito - USFQ).

## Descripción del Proyecto

El corazón de este repositorio es el cuaderno interactivo `EDA.ipynb`. Este notebook está diseñado como una **clase guiada interactiva** que entrelaza conceptos teóricos fundamentales del análisis estadístico con su implementación en código usando **Python, Pandas, Matplotlib y Seaborn**.

El objetivo es enseñar que el EDA no es un paso opcional, sino el "contrato de confianza" que establecemos con nuestros datos antes de cualquier modelamiento.

## Contenidos del Repositorio

- `EDA.ipynb`: Cuaderno principal estructurado metodológicamente. Incluye toda la teoría estadística y lógica de preprocesamiento, acompañada paso a paso del código de análisis.
- `datos/`: Directorio que contiene los datasets relacionales utilizados en el curso (basado en el dataset público de *Instacart*).
  - `instacart_orders.csv`: Histórico de órdenes de los clientes.
  - `products.csv`: Maestro de productos.
  - `order_products.csv`: Detalle transaccional de productos por orden.
  - `aisles.csv` y `departments.csv`: Categorización secundaria de los productos.

## Temas Cubiertos en el Notebook

Durante la lección se abordan conceptualmente y de manera práctica los siguientes temas clave:

1. **Las 4 Dimensiones de la Calidad de los Datos**
   - **Completitud:** Diagnóstico de valores ausentes (Mecanismos MCAR, MAR, MNAR) e imputación vs. eliminación.
   - **Precisión:** Identificación y limpieza de duplicados explícitos y lógicos (tricky duplicates).
   - **Sensibilidad:** Detección de *outliers* usando Criterios de Tukey (IQR/Boxplots) y Regla Empírica (Z-scores).
   - **Consistencia:** Correspondencia entre el tipo de variable estadístico y el tipo computacional (tipos de datos en Pandas).

2. **Técnicas de Muestreo y Granularidad**
   - Tipos de muestreo aleatorio y sesgo posicional.
   - Entendiendo el impacto de la unidad de análisis (Granularidad) en estadísticos y duplicidad funcional.

3. **Análisis Estadístico Univariado y Bivariado**
   - Medidas de Tendencia Central (Media, Mediana, Moda) y casos de uso (Ej: Asimetrías prolongadas).
   - Medidas de Variabilidad e Histogramas / Diagramas KDE.
   - Tests de Normalidad (Gráficos Q-Q).
   - Análisis Multivariado: Covarianza, Correlación (Pearson) y su visualización mediante Scatter Plots e integraciones estructurales (One Big Table).

## Cómo usar este repositorio

Para sacar el mayor provecho al material:

1. Clona el repositorio a tu máquina local.
2. Asegúrate de tener un entorno virtual configurado con:
   - `python >= 3.8`
   - `pandas`
   - `matplotlib`
   - `seaborn`
   - `jupyterlab` o `notebook`
3. Ejecuta `jupyter lab` en tu consola y abre `EDA.ipynb`.
4. Sigue la lectura guiada analizando los bloques teóricos de Markdown y ejecutando progresivamente el código en las celdas inferiores para comprobar la teoría expuesta.
