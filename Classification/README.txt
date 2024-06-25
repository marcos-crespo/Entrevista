El análisis completo de los datos se encuentra en Classification.ipynb o Classification.pdf.

Para realizar predicciones hemos dejado unos archivos de prueba. 

En una terminal de Python:

1. Comprobación de requisitos:

instalar las dependencias necesarias con:

	pip install -r requirements.txt

2. Asegúrate que el script 'productionFile.py' y el modelo 'logistic_regression_model.pkl' están en la misma carpeta y que tu terminal está en la misma.

3. Ejecuta el script 'productionFile.py' dando como argumentos los nombres del modelo y del script de salida (HACEN FALTA LAS EXTENSIONES DE ARCHIVO) con:
	python productionFile.py <input_csv> <output_csv>

4. Las predicciones se habrán generado en la carpeta donde estaba el script
