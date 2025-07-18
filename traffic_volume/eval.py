import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from attention_layer import AttentionLayer
from traffic_dataset import TrafficDataLoader

def main(model_path, data_path, result_dir):
    print("Loading test data...")
    data_loader = TrafficDataLoader(data_path=data_path)
    test_X, test_Y = data_loader.test_X, data_loader.test_Y

    print(f"Loading model from '{model_path}'...")
    best_model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
    
    print("Evaluating model performance...")
    test_loss = best_model.evaluate(test_X, test_Y, verbose=0)
    print(f"\n📈 Final Test MSE: {test_loss:.4f}\n")

    print("Generating predictions...")
    predictions = best_model.predict(test_X)

    run_name = os.path.basename(model_path).replace('_best.keras', '')
    results_csv_path = os.path.join(result_dir, f'test_results_{run_name}.csv')
    num_features = test_Y.shape[1]
    results_df = pd.DataFrame(np.concatenate([test_Y, predictions], axis=1), 
                              columns=[f'true_{i+1}' for i in range(num_features)] + [f'pred_{i+1}' for i in range(num_features)])
    results_df.to_csv(results_csv_path, index=False)
    print(f"✅ Results saved to '{results_csv_path}'.")

    plot_dir = os.path.join(result_dir, 'test_plots', run_name)
    os.makedirs(plot_dir, exist_ok=True)
    for i in range(num_features):
        plt.figure(figsize=(15, 6))
        plt.plot(test_Y[:, i], label='True Value', color='blue')
        plt.plot(predictions[:, i], label='Predicted Value', color='red', linestyle='--')
        plt.title(f'Feature {i+1} Comparison for {run_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'feature_{i+1}_trend.png'))
        plt.close()

    print(f"✅ All comparison plots saved in '{plot_dir}' directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained traffic prediction model.')
    parser.add_argument('--model_path', required=True, type=str, help='Path to the trained .keras model file.')
    parser.add_argument('--data_dir', default='./datasets', type=str, help='Dataset directory.')
    parser.add_argument('--result_dir', default='./results', type=str, help='Results directory.')
    args = parser.parse_args()
    
    main(args.model_path, args.data_dir, args.result_dir)