import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from datasets.speed_dataset import SpeedDataLoader

def plot_overall_scatter(true_values, predictions, filename, result_dir):
    """
    전체 테스트 데이터에 대한 실제 값과 예측 값을 비교하는 산점도를 생성합니다.
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(true_values, predictions, alpha=0.3, label='예측값')
    # 완벽한 예측을 나타내는 y=x 선 추가
    perfect_line = np.linspace(min(true_values.min(), predictions.min()), 
                               max(true_values.max(), predictions.max()), 100)
    plt.plot(perfect_line, perfect_line, 'r--', label='완벽한 예측 (y=x)')
    plt.title(f'전체 예측 결과 비교 (산점도): {filename}')
    plt.xlabel('실제 값 (True Values)')
    plt.ylabel('예측 값 (Predicted Values)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    
    save_path = os.path.join(result_dir, f'overall_scatter_plot_{filename}.png')
    plt.savefig(save_path)
    print(f"✅ 전체 비교 산점도가 '{save_path}'에 저장되었습니다.")
    plt.close()


def main(model_path, data_path, result_dir):
    """
    훈련된 모델을 평가하고, 예측 결과를 CSV와 그래프로 저장하며, 주요 데이터를 터미널에 출력합니다.
    """
    # --- 1. 테스트 데이터 로드 ---
    print("테스트 데이터를 로딩합니다...")
    data_loader = SpeedDataLoader(data_path=data_path)
    test_X, test_Y = data_loader.test_X, data_loader.test_Y

    # --- 2. 훈련된 모델 로드 ---
    if not os.path.exists(model_path):
        print(f"❌ 오류: '{model_path}'에서 모델을 찾을 수 없습니다.")
        return
        
    print(f"'{model_path}'에서 모델을 로딩합니다...")
    best_model = load_model(model_path)
    best_model.summary()
    
    # --- 3. 전체 모델 성능 평가 (MSE) ---
    print("\n모델 성능을 평가합니다...")
    overall_test_loss = best_model.evaluate(test_X, test_Y, verbose=0)
    print(f"📈 전체 테스트 데이터 MSE: {overall_test_loss:.4f}")

    # --- 4. 예측 생성 ---
    print("예측값을 생성합니다...")
    predictions = best_model.predict(test_X)

    run_name = os.path.basename(model_path).replace('_best.keras', '')
    os.makedirs(result_dir, exist_ok=True)
    
    results_csv_path = os.path.join(result_dir, f'test_results_{run_name}.csv')
    num_features = test_Y.shape[1] if len(test_Y.shape) > 1 else 1
    
    if num_features == 1:
        test_Y_reshaped = test_Y.reshape(-1, 1)
        predictions_reshaped = predictions.reshape(-1, 1)
    else:
        test_Y_reshaped = test_Y
        predictions_reshaped = predictions

    results_df = pd.DataFrame(np.concatenate([test_Y_reshaped, predictions_reshaped], axis=1), 
                              columns=[f'true_{i+1}' for i in range(num_features)] + [f'pred_{i+1}' for i in range(num_features)])
    
    # --- 5. 방면 정보 추가 및 방면별 성능 평가 ---
    if test_X.shape[1] > 0:
        directions = test_X[:, -1].astype(int)
        results_df['direction'] = directions
    else:
        results_df['direction'] = -1

    results_df.to_csv(results_csv_path, index=False)
    print(f"✅ 결과가 '{results_csv_path}'에 저장되었습니다. (방면 정보 포함)")

    # --- 5-1. 각 방면별 MSE 계산 및 출력 ---
    print("\n--- 🛣️ 방면별 MSE 결과 ---")
    unique_directions = sorted(results_df['direction'].unique())
    for direction_code in unique_directions:
        if direction_code == -1: continue
        
        direction_df = results_df[results_df['direction'] == direction_code]
        true_vals = direction_df[[f'true_{i+1}' for i in range(num_features)]].values
        pred_vals = direction_df[[f'pred_{i+1}' for i in range(num_features)]].values
        
        directional_mse = mean_squared_error(true_vals, pred_vals)
        print(f"  [방면 코드 {direction_code}] MSE: {directional_mse:.4f}")

    # --- 5-2. 방면별 전체 데이터 출력 ---
    print("\n--- 🛣️ 방면별 전체 데이터 예측 결과 ---")
    for direction_code in unique_directions:
        if direction_code == -1: continue
        print(f"\n[방면 코드: {direction_code}]")
        direction_df = results_df[results_df['direction'] == direction_code]
        display_df = direction_df.drop(columns=['direction'])
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(display_df.to_string(index=False))
    
    # --- 6. 그래프 생성 및 저장 ---
    plot_dir = os.path.join(result_dir, 'test_plots', run_name)
    os.makedirs(plot_dir, exist_ok=True)

    # 6-1. 전체 비교 산점도 그래프
    plot_overall_scatter(test_Y_reshaped, predictions_reshaped, run_name, result_dir)
    
    # 6-2. 출력 피처별 시계열 그래프 (기존 기능)
    for i in range(num_features):
        plt.figure(figsize=(15, 6))
        plt.plot(test_Y_reshaped[:, i], label='실제 값 (True Value)', color='blue')
        plt.plot(predictions_reshaped[:, i], label='예측 값 (Predicted Value)', color='red', linestyle='--')
        plt.title(f'출력 피처 {i+1} 전체 시계열 비교: {run_name}')
        plt.xlabel("Time Step")
        plt.ylabel("Speed")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'feature_{i+1}_overall_trend.png'))
        plt.close()
    print(f"\n✅ 전체 피처별 비교 그래프가 '{plot_dir}'에 저장되었습니다.")

    # 6-3. 방면별 시계열 그래프
    directional_plot_dir = os.path.join(plot_dir, 'directional_plots')
    os.makedirs(directional_plot_dir, exist_ok=True)
    
    for direction_code in unique_directions:
        if direction_code == -1: continue
        
        direction_df = results_df[results_df['direction'] == direction_code]
        
        for i in range(num_features):
            true_vals = direction_df[f'true_{i+1}'].values
            pred_vals = direction_df[f'pred_{i+1}'].values
            
            plt.figure(figsize=(15, 6))
            plt.plot(true_vals, label='실제 값 (True Value)', color='blue')
            plt.plot(pred_vals, label='예측 값 (Predicted Value)', color='red', linestyle='--')
            plt.title(f'[방면 {direction_code}] 피처 {i+1} 시계열 비교: {run_name}')
            plt.xlabel("Time Step (for this direction)")
            plt.ylabel("Speed")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(directional_plot_dir, f'direction_{direction_code}_feature_{i+1}_trend.png'))
            plt.close()

    print(f"✅ 모든 방면별 비교 그래프가 '{directional_plot_dir}'에 저장되었습니다.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='훈련된 교통 예측 모델을 평가합니다.')
    parser.add_argument('--model_path', required=True, type=str, help='훈련된 .keras 모델 파일의 경로.')
    parser.add_argument('--data_dir', default='./datasets', type=str, help='데이터셋 디렉토리.')
    parser.add_argument('--result_dir', default='./results', type=str, help='결과 저장 디렉토리.')
    args = parser.parse_args()
    
    main(args.model_path, args.data_dir, args.result_dir)
