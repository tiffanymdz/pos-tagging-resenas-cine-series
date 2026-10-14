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



#USOS DE LA IA Felipe Montenegro Artavia 

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
