import argparse
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model

def predict_new_data(model_path, scaler_path, input_data):
    """
    Predict traffic speed using a trained model and scaler.

    Args:
        model_path (str): Path to the saved Keras model file (.keras).
        scaler_path (str): Path to the saved Scaler file (.pkl).
        input_data (np.ndarray): New input data for prediction (2D array).

    Returns:
        np.ndarray: Predicted traffic speeds.
    """

    # Check if model and scaler files exist
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at '{model_path}'")
        return
    if not os.path.exists(scaler_path):
        print(f"❌ Error: Scaler file not found at '{scaler_path}'")
        return

    print("--- Loading model and scaler ---")
    try:
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Model and scaler loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    # Validate input data shape
    if input_data.ndim != 2:
        print(f"❌ Error: Input data must be a 2D array, but got {input_data.ndim} dimensions.")
        return

    print("\n--- Predicting ---")
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Generate predictions
    prediction = model.predict(input_data_scaled)
    
    return prediction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict traffic speed using a trained model.')
    # Note: Default paths can be changed as needed
    # Checkpoint paths for model and scaler
    parser.add_argument('--model_path', default='./checkpoint/model_dnn_lr0.001_seed9999_best.keras', type=str, help='Path to the trained Keras model file.')
    parser.add_argument('--scaler_path', default='./checkpoint/scaler_dnn_lr0.001_seed9999.pkl', type=str, help='Path to the trained scaler file.')
    args = parser.parse_args()

    # Define sample input data for prediction (24 features + 1 direction)
    sample_input = np.array([[
      122,177,	234,	145,	100,	61,	197,	30,	151,	77,	22,	173,	50,	117,	193,	70,	0,	24,	31,	20,	17,	17,	16,	
      1
    ]]) # shape: (1, 24)

    # Perform prediction
    predicted_value = predict_new_data(args.model_path, args.scaler_path, sample_input)

    if predicted_value is not None:
        print("\n--- Prediction Result ---")
        print(f"Input Data:\n{sample_input[0]}")
        print(f"🚀 Predicted Speed: {predicted_value[0][0]:.2f}")
