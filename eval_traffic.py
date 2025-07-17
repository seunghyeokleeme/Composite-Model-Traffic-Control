import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from attention_layer import AttentionLayer
from traffic_dataset import TrafficDataLoader

def main(model_path: str, data_path: str, save_dir: str):
    print("Loading test data...")
    try:
        data_loader = TrafficDataLoader(data_path=data_path)
        test_X, test_Y = data_loader.test_X, data_loader.test_Y
        print("✅ Test data loaded.")
    except FileNotFoundError:
        print(f"❌ Error: Data not found at '{data_path}'. Please check the path.")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    print(f"Loading model from '{model_path}'...")
    try:
        best_model = load_model(
            model_path,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        print("✅ Model loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at '{model_path}'. Please check the path.")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print("Evaluating model performance on the test set...")
    test_loss = best_model.evaluate(test_X, test_Y, verbose=0)
    print(f"\n📈 Final Test MSE (Mean Squared Error): {test_loss:.4f}\n")

    print("Generating predictions for the entire test set...")
    predictions = best_model.predict(test_X)
    print("✅ Predictions generated.")

    num_features = test_Y.shape[1]
    true_cols = [f'true_feature_{i+1}' for i in range(num_features)]
    pred_cols = [f'pred_feature_{i+1}' for i in range(num_features)]
    
    results_df = pd.DataFrame(np.concatenate([test_Y, predictions], axis=1), columns=true_cols + pred_cols)
    results_df.to_csv(os.path.join(save_dir, 'test_results.csv'), index=False)
    print("✅ True values and predictions saved to 'test_results.csv'.")

    print("Generating and saving comparison plots...")
    
    for i in range(num_features):
        plt.figure(figsize=(15, 6))
        plt.plot(test_Y[:, i], label='True Value', color='blue', alpha=0.7)
        plt.plot(predictions[:, i], label='Predicted Value', color='red', linestyle='--')
        plt.title(f'Feature {i+1}: True vs. Predicted Trend')
        plt.xlabel('Test Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'feature_{i+1}_trend_comparison.png'))
        plt.close()

    print(f"✅ All comparison plots saved in '{save_dir}' directory.")


if __name__ == '__main__':
    MODEL_PATH = 'model_B_best.h5'
    DATA_PATH = 'data/traffic_volume'
    SaveDir = 'output'

    main(model_path=MODEL_PATH, data_path=DATA_PATH, save_dir=SaveDir)
