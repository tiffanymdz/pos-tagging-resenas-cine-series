# CineScope — Dashboard de análisis semántico

Dashboard interactivo hecho en **Plotly Dash**, con corpus real de 12.368 reseñas y análisis de BETO.

## Ejecución en Windows

1. Descomprimí esta carpeta.
2. Abrí una terminal en `CineScope_Dashboard`.
3. Instalá las dependencias: `python -m pip install -r requirements.txt`
4. Iniciá la aplicación: `python dashboard/app.py`
5. Abrí `http://127.0.0.1:8050` en el navegador.

No requiere MongoDB, GPU ni descargar modelos para explorar los resultados precalculados.

## Agregar las métricas reales de comparación

Ejecutá `05_comparacion_final.ipynb` y copiá `resumen_comparacion.csv` a `dashboard/data/resumen_comparacion.csv`. Reiniciá la aplicación. La sección de comparación mostrará las métricas reales automáticamente.

## Contenido

- Panorama del corpus y filtros por género, polaridad, origen y texto.
- t-SNE interactivo precalculado para BoW, TF-IDF, CBOW, Skip-Gram y BETO, sobre una misma muestra de 550 reseñas.
- Búsquedas semánticas BETO previamente calculadas; **no** acepta consultas nuevas en tiempo real sin desplegar el modelo.
- Polisemia contextual y predicciones Masked LM.
- Comparación cuantitativa cuando se adjunta el CSV de métricas.

## Consideraciones metodológicas

Las proyecciones t-SNE se generaron por separado para cada representación, sobre los mismos 550 IDs y con la misma semilla. No comparar distancias absolutas entre paneles. Las visualizaciones son exploratorias. Las métricas no se inventan. El CSV `resenas.csv.gz` contiene textos y metadatos del corpus compartido por el equipo.

La rúbrica también solicita clasificación por **género** y análisis por **origen**, además de polaridad. Si el notebook 05 solo calculó polaridad, esas evaluaciones deben añadirse al análisis antes de dar por terminado el proyecto.
