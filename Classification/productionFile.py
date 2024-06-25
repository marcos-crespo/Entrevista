import joblib
import numpy as np
import pandas as pd
import sys

# Cargar el modelo
model = joblib.load('logistic_regression_model.pkl')

def check_input_data(input_data):
    """
    Verifica que los datos de entrada tengan 30 columnas y que todas sean numéricas.
    
    :param input_data: numpy array de las características a predecir
    :return: None
    :raises ValueError: si las verificaciones fallan
    """
    if input_data.shape[1] != 30:
        raise ValueError("Los datos de entrada deben tener 30 columnas.")
    if not np.issubdtype(input_data.dtype, np.number):
        raise ValueError("Todas las columnas deben ser numéricas.")
    
    # Asegúrate de que input_data sea un numpy array
    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data)

def predict(input_data):
    """
    Realiza predicciones utilizando el modelo entrenado.
    
    :param input_data: numpy array de las características a predecir
    :return: predicción del modelo
    """
    # Verificar los datos de entrada
    check_input_data(input_data)
    
    # Realizar la predicción
    prediction = model.predict(input_data)
    return prediction

def save_predictions(predictions, output_file):
    """
    Guarda las predicciones en un archivo CSV.
    
    :param predictions: array de predicciones
    :param output_file: nombre del archivo de salida
    :return: None
    """
    df = pd.DataFrame(predictions, columns=['Prediction'])
    df.to_csv(output_file, index=False)

# Ejemplo de uso
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python predict.py <input_csv> <output_csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Cargar datos de entrada desde un archivo CSV
    input_data = pd.read_csv(input_file).values

    # Realizar la predicción
    try:
        predictions = predict(input_data)
        save_predictions(predictions, output_file)
        print(f'Predicciones guardadas en {output_file}')
    except ValueError as e:
        print(f'Error: {e}')
