import argparse
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import matplotlib as mat
import matplotlib.pyplot as plt

from speed_dataset import SpeedDataLoader

def set_publication_style():
    mat.rc('font', family='DejaVu Serif')
    mat.rcParams['mathtext.fontset'] = 'dejavuserif'
    mat.rcParams['font.size'] = 16           
    mat.rcParams['axes.labelsize'] = 16      
    mat.rcParams['xtick.labelsize'] = 14     
    mat.rcParams['ytick.labelsize'] = 14     
    mat.rcParams['legend.fontsize'] = 14     
    # mat.rcParams['axes.titlesize'] = 12    
    mat.rcParams['savefig.dpi'] = 1200       
    mat.rcParams['figure.dpi'] = 1200
    mat.rcParams['lines.linewidth'] = 1.5
    mat.rcParams['axes.grid'] = True     
    mat.rcParams['axes.unicode_minus'] = False 
    mat.rcParams['axes.xmargin'] = 0.1
    mat.rcParams['axes.ymargin'] = 0.1

def plot_overall_scatter(true_values, predictions, filename, result_dir):
    """
    Generates a scatter plot comparing true values and predictions for the entire test dataset.
    """
    set_publication_style()
    plt.figure(figsize=(6, 6))
    plt.scatter(true_values, predictions, alpha=0.3, edgecolors='k', s=40, facecolors='royalblue')
    perfect_line = np.linspace(min(true_values.min(), predictions.min()), 
                               max(true_values.max(), predictions.max()), 100)
    plt.plot(perfect_line, perfect_line, 'r--', label='y=x (Perfect Prediction)')
    # plt.title(f'Overall Prediction vs. True Values (Scatter Plot): {filename}')
    plt.xlabel('Actual Speed (km/h)')
    plt.ylabel('Predicted Speed (km/h)')
    plt.legend(loc='best')
    plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    save_path = os.path.join(result_dir, f'overall_scatter_plot_{filename}.eps')
    plt.savefig(save_path, format='eps', bbox_inches='tight')
    print(f"✅ Overall scatter plot saved to '{save_path}'")
    plt.close()

def plot_directional_trends(results_df, result_dir, run_name):
    """
    Generates time-series graphs comparing true and predicted values for each direction.
    """
    set_publication_style()
    directional_plot_dir = os.path.join(result_dir, 'test_plots', run_name, 'directional_plots')
    os.makedirs(directional_plot_dir, exist_ok=True)
    
    unique_directions = sorted(results_df['direction'].unique())
    # directions_map = ["Busan National University", "Busan City Hall", "World Cup", "Silli", "Allak", "Yeonsan Tunnel"]

    for direction_code in unique_directions:
        if direction_code == -1: continue
        
        direction_df = results_df[results_df['direction'] == direction_code]
        
        true_vals = direction_df['true_speed'].values
        pred_vals = direction_df['pred_speed'].values
        
        plt.figure(figsize=(15,6))
        
        plt.ylim(0, 50)
        if direction_code == 0:
          plt.plot(true_vals, label='Actual', color='royalblue')
          plt.plot(pred_vals, label='Predicted', color='red', linestyle='--')
        else:
          plt.plot(true_vals, color='royalblue')
          plt.plot(pred_vals, color='red', linestyle='--')
        
        plt.xlabel("Time Step")
        plt.ylabel("Speed (kph)")
        if direction_code == 0:
          plt.legend(loc='best')
        plt.grid(True, linestyle=':')
        plt.savefig(os.path.join(directional_plot_dir, f'direction_{direction_code}_trend.eps'), format='eps', bbox_inches='tight')
        plt.close()

    print(f"✅ All directional comparison plots saved to '{directional_plot_dir}'")

def plot_all_directions_combined(results_df, result_dir, run_name):
    """
    ✅ ADDED: Generates a single graph containing all 6 true values and 6 predicted values.
    """
    set_publication_style()
    plot_dir = os.path.join(result_dir, 'test_plots', run_name)
    plt.figure(figsize=(15, 8))
    
    unique_directions = sorted(results_df['direction'].unique())
    # directions_map = ["Busan National University", "Busan City Hall", "World Cup", "Silli", "Allak", "Yeonsan Tunnel"]
    
    # Use a color map to assign a unique color to each direction
    colors = plt.cm.get_cmap('tab10', len(unique_directions))

    for direction_code in unique_directions:
        if direction_code == -1: continue
        
        direction_df = results_df[results_df['direction'] == direction_code]
        
        true_vals = direction_df['true_speed'].values
        pred_vals = direction_df['pred_speed'].values
        
        # try:
        #     direction_name = directions_map[direction_code]
        # except IndexError:
        #     direction_name = f"Direction {direction_code}"

        # Plot true values with a solid line and predicted values with a dashed line
        plt.plot(true_vals, color=colors(direction_code), linestyle='-', label=f'Approach {direction_code} - Actual')
        plt.plot(pred_vals, color=colors(direction_code), linestyle='--', label=f'Approach {direction_code} - Predicted')

    # plt.title(f'Combined Time Series Comparison for All Directions: {run_name}')
    plt.xlabel("Time Step (Test Period: Oct 16-31)")
    plt.ylabel("Speed (kph)")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    
    save_path = os.path.join(plot_dir, 'all_directions_combined_trend.eps')
    plt.savefig(save_path, format='eps', bbox_inches='tight')
    print(f"✅ Combined plot for all directions saved to '{save_path}'")
    plt.close()

def main(model_path, scaler_path, data_path, result_dir):
    """
    Evaluates the trained model, saves prediction results to CSV and graphs, and prints key data to the terminal.
    """
    # --- 1. Load test data ---
    print("Loading test data...")
    data_loader = SpeedDataLoader(data_path=data_path)
    test_X, test_Y = data_loader.test_X, data_loader.test_Y

    # --- 2. Load trained model and scaler ---
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at '{model_path}'")
        return
    if not os.path.exists(scaler_path):
        print(f"❌ Error: Scaler not found at '{scaler_path}'")
        return
        
    print(f"Loading model from '{model_path}'...")
    best_model = load_model(model_path)
    best_model.summary()
    
    print(f"Loading scaler from '{scaler_path}'...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    test_X_scaled = scaler.transform(test_X)
    print("✅ Test data scaled successfully.")

    # --- 3. Evaluate overall model performance (MSE) ---
    print("\nEvaluating model performance...")
    overall_test_loss = best_model.evaluate(test_X_scaled, test_Y, verbose=0)
    print(f"📈 Overall Test Data MSE: {overall_test_loss:.4f}")

    # --- 4. Generate predictions ---
    print("Generating predictions...")
    predictions = best_model.predict(test_X_scaled)

    run_name = os.path.basename(model_path).replace('model_', '').replace('_best.keras', '')
    os.makedirs(result_dir, exist_ok=True)
    
    results_df = pd.DataFrame({
        'true_speed': test_Y.flatten(),
        'pred_speed': predictions.flatten()
    })
    
    # --- 5. Add direction info and evaluate performance by direction ---
    if test_X.shape[1] > 0:
        results_df['direction'] = test_X[:, -1].astype(int)
    else:
        results_df['direction'] = -1

    results_csv_path = os.path.join(result_dir, f'test_results_{run_name}.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"✅ Results saved to '{results_csv_path}' (including direction info)")

    print("\n--- 🛣️ MSE by Direction ---")
    for direction_code in sorted(results_df['direction'].unique()):
        if direction_code == -1: continue
        direction_df = results_df[results_df['direction'] == direction_code]
        directional_mse = mean_squared_error(direction_df['true_speed'], direction_df['pred_speed'])
        print(f"  [Direction Code {direction_code}] MSE: {directional_mse:.4f}")

    # --- 6. Generate and save plots ---
    plot_dir = os.path.join(result_dir, 'test_plots', run_name)
    os.makedirs(plot_dir, exist_ok=True)

    plot_overall_scatter(results_df['true_speed'], results_df['pred_speed'], run_name, result_dir)
    plot_directional_trends(results_df, result_dir, run_name)
    plot_all_directions_combined(results_df, result_dir, run_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained traffic speed prediction model.')
    parser.add_argument('--model_path', required=True, type=str, help='Path to the trained .keras model file.')
    parser.add_argument('--scaler_path', required=True, type=str, help='Path to the trained .pkl scaler file.')
    parser.add_argument('--data_dir', default='./datasets', type=str, help='Dataset directory.')
    parser.add_argument('--result_dir', default='./results', type=str, help='Directory to save results.')
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    
    main(args.model_path, args.scaler_path, args.data_dir, args.result_dir)
