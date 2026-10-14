Herramienta: Claude (Anthropic).

¿Cómo nos ayudó la IA?

El uso de la inteligencia artificial fue de gran ayuda para el desarrollo de nuestro proyecto,
ya que nos permitió profundizar un poco más en el tema, orientarnos en la asignación de etiquetas,
comprender mejor las comparaciones realizadas y elegir los gráficos más adecuados para representar
e interpretar los resultados.
Además, nos sirvió como una herramienta de apoyo para resolver dudas y guiarnos durante el proceso de desarrollo 
del proyecto.

¿Qué hicimos nosotros?
Nos encargamos de desarrollar el proyecto, trabajar con los datos, ejecutar y revisar los resultados, analizar las etiquetas y 
realizar los ajustes necesarios. También subimos el cuaderno y los resultados al repositorio de GitHub.



#USOS DE LA IA

#Qué herramientas de IA utilizaron y para qué tareas específicas.

Se utilzó Claude (Anthropic)
en problemas con los datos, en el diseño del muestreo y en el eje c ayudó a calcular metricas y graficos.

#Ejemplos de prompts utilizados (mínimo 3 a 5 ejemplos representativos). 
Detección de un problema en los datos

Tengo una consulta sobre el archivo CSV después del proceso de limpieza de datos. He notado que algunas reseñas escritas en español parecen estar incompletas o cortadas, ya que ciertas frases terminan de forma repetina o no tienen sentido completo.

Ayúdame a investigar este problema en el conjunto de datos.

Cómo asegurarme de un resultado

Necesito verificar la calidad de las traducciones de las reseñas de mi archivo CSV limpio. ¿Cómo puedo asegurarme de que las traducciones al español estén completas y que no existan reseñas cortadas, incompletas o que hayan perdido parte de su contenido durante el proceso de traducción o limpieza?

Tamaño del corpus y diseño del muestreo

Necesito construir un corpus de 6.000 reseñas para mi proyecto de análisis de texto. Ayúdame a verificar que el archivo CSV limpio cumpla con este requisito y que la selección de las reseñas respete el diseño del muestreo establecido.

El muestreo se organiza de la siguiente manera:

5 géneros cinematográficos × 2 periodos, para un total de 10 grupos o celdas de muestreo.

Una cuota objetivo de 640 reseñas por celda, según el diseño propuesto. Verifica esta cuota, ya que 640 × 10 equivale a 6.400 reseñas y no a 6.000. Propón una distribución coherente que permita alcanzar exactamente las 6.000 reseñas sin alterar los criterios del proyecto.

 #Reflexión sobre cómo la IA ayudó en su aprendizaje (1 a 2 párrafos). 
La IA me ayudó a avanzar en las partes técnicas que no dominaba del todo: cómo muestrear de forma estratificada con semilla fija, cómo traducir un corpus largo en Colab guardando el avance, y cómo comparar dos grupos con una prueba estadística.
Lo más importante que aprendí es que la salida de la IA hay que verificarla.
Usé la IA como apoyo y no como sustituto, si me sirvió para tener mas ideas y complementar con lo que ya tenía. Ya que habian algunas cosas que no quedaban del todo claras y la IA me ayudó a que quedara mas claro.


#Qué modificaciones hicieron al código o al análisis generado por IA. 
Detalles de presentación: pedí quitar la notación científica de los valores p y reordenar la tabla de longitud de reseñas, que tenía celdas vacías en la primera fila.
Reseñas cortadas: decidí eliminarlas del corpus en lugar de volver a traducirlas.

# Uso de IA

## Herramientas utilizadas

- **Claude (Anthropic)**, mediante chat.

## Tareas en las que se usó

- **Revisión de los datos:** análisis del CSV limpio (traducciones que parecían cortadas, oraciones pegadas sin espacio).
- **POS Tagging:** adaptación de la demo del profe, que estaba en inglés, a español con spaCy (`es_core_news_md`) y con un etiquetador de NLTK entrenado con `cess_esp`.
- **Métricas del eje B:** índice de negación e índice de adverbios de intensidad, y el archivo `src/metrics.py`.
- **Depuración de errores:** `ModuleNotFoundError`, `NameError`, límite de tamaño de archivos en GitHub, problemas de rutas.

En todos los casos, la IA propuso código o explicaciones y yo ejecuté todo en mi computadora, paso por paso, mostrando la salida antes de seguir.

## Ejemplo de prompts

 1. > "hay algunas reseñas en español que se cortan a la mitad [...] eso puede afectar el tagging?"

   Resultado: comparó las traducciones con el original en inglés. Las reseñas que yo veía cortadas estaban completas y era Excel recortando la vista, pero encontró oraciones pegadas sin espacio (por ejemplo, "corazón.Ver") en 3.031 reseñas, y propuso una expresión regular para corregirlas.

## Reflexión

La IA me ayudó sobre todo a entender el porqué de cada paso, no solo a obtener código: por qué NLTK necesita entrenarse para español, por qué hay que etiquetar los mismos tokens que spaCy para poder comparar, y por qué las métricas se calculan sin puntuación y por cada 100 tokens. Trabajar paso a paso, corriendo yo cada celda, me permitió encontrar errores concretos, como el modelo equivocado o una columna mal elegida.

## Modificaciones al código

- Reemplacé la demo del profe (en inglés, con `en_core_web_sm` y el `pos_tag` de NLTK) por un flujo en español con `es_core_news_md` y un etiquetador entrenado con `cess_esp`.
- Etiqueté con NLTK **oración por oración**, y no toda la lista de tokens de una vez como indica la guía, para que el contexto del bigrama no pase de una oración a otra. La IA propuso este cambio y yo lo apliqué.
- Guardé el resultado como `pos_analysis.csv.gz` porque GitHub no permite subir por la web archivos de más de 25 MB.
- Reorganicé mis notebooks: quité celdas repetidas, ajusté las rutas a la carpeta `data/` y pasé las funciones de métricas a `src/metrics.py`.
- Verifiqué los resultados: comprobé que `src/metrics.py` da los mismos números que el cálculo del notebook, y revisé a mano ocho reseñas que parecían cortadas.
