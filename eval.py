import argparse
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, LeakyReLU
from tqdm import trange
import matplotlib.pyplot as plt

# 사용자 정의 어텐션 레이어를 불러옵니다.
# 이 코드를 실행하기 전에 'attention_layer.py'가 'traffic_volume' 폴더 안에 있는지 확인해주세요.
from traffic_volume.attention_layer import AttentionLayer

# --- 1. 모델 정의 및 로드 함수 ---

def load_lstm_model(weights_path):
    """
    저장된 가중치를 사용하여 LSTM 모델을 정의하고 불러옵니다.
    traffic_volume/train.py의 모델 구조와 동일해야 합니다.
    """
    input_layer = Input(shape=(24, 17), name="Input_Tensor")
    x = Bidirectional(LSTM(1000, return_sequences=True, activation='tanh'))(input_layer)
    x = Bidirectional(LSTM(500, return_sequences=True, activation='tanh'))(x)
    x = Bidirectional(LSTM(100, return_sequences=True, activation='tanh'))(x)
    x = AttentionLayer(name="Temporal_Attention")(x)
    x = Dense(2048, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(16)(x)
    model = Model(inputs=input_layer, outputs=output_layer, name="Deep_Attention_Model")
    
    model.load_weights(weights_path)
    print("LSTM 모델 로드 완료.")
    return model

def load_dnn_model(weights_path):
    """
    저장된 가중치를 사용하여 DNN 모델을 정의하고 불러옵니다.
    traffic_speed/train.py의 모델 구조와 동일해야 합니다.
    """
    model = Sequential()
    model.add(Dense(64, input_shape=(24,)))
    model.add(LeakyReLU(alpha=0.02))
    for _ in range(15):
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.02))
    model.add(Dense(1, activation='linear'))
    
    model.load_weights(weights_path)
    print("DNN 모델 로드 완료.")
    return model

# --- 2. 신호 최적화 프레임워크 클래스 ---

class Signal_Framework:
  def __init__(self, lstm_model, dnn_model):
    self.min_signal_times = np.array([10, 31, 15, 11, 8, 15])
    self.max_signal_times = np.array([45-4, 55-4, 45-4, 40-4, 35-4, 40-4])

    self.lstm_model = lstm_model
    self.dnn_model = dnn_model
    self.lstm_input = None
    self.lstm_output = None
    self.best_solution = None
    self.best_fitness_per_generation = []

  def _prepare_dnn_input(self, signal_values):
      holiday_info = self.lstm_input[0, -1, 16]
      base_features = np.concatenate([self.lstm_output.flatten(), [holiday_info], signal_values])
      dnn_input = np.zeros((6, 24))
      for i in range(6):
          dnn_input[i, :] = np.concatenate([base_features, [i]])
      return dnn_input

  def evaluate_population(self, population):
      all_dnn_inputs = []
      valid_indices = []
      for i, individual in enumerate(population):
          if np.sum(individual) <= 200:
              prepared_data = self._prepare_dnn_input(individual)
              all_dnn_inputs.append(prepared_data)
              valid_indices.append(i)
      
      if not all_dnn_inputs:
          return np.zeros(len(population))

      all_dnn_inputs_combined = np.vstack(all_dnn_inputs)
      all_predicted_speeds = self.dnn_model.predict(all_dnn_inputs_combined, verbose=0)
      
      fitness_scores = np.zeros(len(population))
      predicted_volumes = self.lstm_output.flatten()
      weights = np.array([
          np.sum(predicted_volumes[0:2]), np.sum(predicted_volumes[2:5]),
          np.sum(predicted_volumes[5:8]), np.sum(predicted_volumes[8:11]),
          np.sum(predicted_volumes[11:13]), np.sum(predicted_volumes[13:16])
      ])
      total_volume = np.sum(predicted_volumes)
      if total_volume == 0: total_volume = 1

      start_idx = 0
      for i, original_idx in enumerate(valid_indices):
          speeds_for_individual = all_predicted_speeds[start_idx : start_idx + 6].flatten()
          weighted_avg_speed = np.sum(speeds_for_individual * weights) / total_volume
          fitness_scores[original_idx] = weighted_avg_speed
          start_idx += 6
      return fitness_scores

  def evaluate_signal(self, signal_values):
      """[추가됨] 단일 신호 조합의 성능(가중 평균 속도)을 평가하는 함수"""
      dnn_input = self._prepare_dnn_input(signal_values)
      predicted_speeds = self.dnn_model.predict(dnn_input, verbose=0).flatten()
      
      predicted_volumes = self.lstm_output.flatten()
      weights = np.array([
          np.sum(predicted_volumes[0:2]), np.sum(predicted_volumes[2:5]),
          np.sum(predicted_volumes[5:8]), np.sum(predicted_volumes[8:11]),
          np.sum(predicted_volumes[11:13]), np.sum(predicted_volumes[13:16])
      ])
      total_volume = np.sum(predicted_volumes)
      if total_volume == 0: total_volume = 1

      weighted_avg_speed = np.sum(predicted_speeds * weights) / total_volume
      return weighted_avg_speed

  def select_parents(self, population, fitness, num_parents):
      parents = np.empty((num_parents, population.shape[1]), dtype=np.int32)
      temp_fitness = np.copy(fitness)
      for parent_num in range(num_parents):
          max_fitness_idx = np.argmax(temp_fitness)
          parents[parent_num, :] = population[max_fitness_idx, :]
          temp_fitness[max_fitness_idx] = -999999
      return parents

  def crossover(self, parents, offspring_size):
      offspring = np.empty(offspring_size, dtype=np.int32)
      crossover_point = offspring_size[1] // 2
      for k in range(offspring_size[0]):
          parent1_idx = k % parents.shape[0]
          parent2_idx = (k + 1) % parents.shape[0]
          offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
          offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
      return offspring

  def mutate(self, offspring_crossover, mutation_rate):
      for idx in range(offspring_crossover.shape[0]):
          if np.random.rand() < mutation_rate:
              random_index = np.random.randint(0, offspring_crossover.shape[1])
              random_value = np.random.randint(-5, 6)
              mutated_gene = offspring_crossover[idx, random_index] + random_value
              offspring_crossover[idx, random_index] = np.clip(
                  mutated_gene, self.min_signal_times[random_index], self.max_signal_times[random_index]
              )
      return offspring_crossover

  def create_initial_population(self, population_size):
      population = np.zeros((population_size, len(self.min_signal_times)), dtype=np.int32)
      for i in range(population_size):
          for j in range(len(self.min_signal_times)):
              population[i, j] = np.random.randint(self.min_signal_times[j], self.max_signal_times[j] + 1)
      return population

  def optimize_signal_times(self, lstm_input, population_size=100, num_generations=200, num_parents_mating=10, mutation_rate=0.3):
      # 1. LSTM으로 미래 교통량 예측
      self.lstm_input = lstm_input
      self.lstm_output = self.lstm_model.predict(self.lstm_input, verbose=0)
      # [수정됨] LSTM 예측 결과를 보기 쉽게 출력
      print("\n--- LSTM 예측 결과 (16개 방향 교통량) ---")
      print(np.round(self.lstm_output).astype(int))
      print("-" * 40)

      # 2. 유전 알고리즘 실행
      self.best_fitness_per_generation = []
      population = self.create_initial_population(population_size)
      print("유전 알고리즘 최적화 시작...")
      for generation in trange(num_generations):
          fitness = self.evaluate_population(population)
          self.best_fitness_per_generation.append(np.max(fitness))
          parents = self.select_parents(population, fitness, num_parents_mating)
          offspring_size = (population_size - parents.shape[0], population.shape[1])
          offspring_crossover = self.crossover(parents, offspring_size)
          offspring_mutation = self.mutate(offspring_crossover, mutation_rate)
          population[0:parents.shape[0], :] = parents
          population[parents.shape[0]:, :] = offspring_mutation
      
      final_fitness = self.evaluate_population(population)
      best_match_idx = np.argmax(final_fitness)
      self.best_solution = population[best_match_idx, :]
      best_solution_fitness = final_fitness[best_match_idx]
      print("최적화 완료.")
      return self.best_solution, best_solution_fitness

  def plot_fitness_trend(self):
      plt.figure(figsize=(10, 6))
      plt.plot(self.best_fitness_per_generation)
      plt.title('Generation vs. Best Fitness Trend')
      plt.xlabel('Generation')
      plt.ylabel('Best Fitness (Weighted Average Speed)')
      plt.grid(True)
      output_filename = 'fitness_trend.png'
      plt.savefig(output_filename, dpi=300)
      print(f"Convergence graph saved to '{output_filename}'")
      plt.show()

# --- 3. 메인 실행 블록 ---

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')

    parser = argparse.ArgumentParser(description='Optimize traffic signal timings using LSTM and DNN models.')
    parser.add_argument('--lstm_weights', type=str, required=True, help='Path to the LSTM model weights file.')
    parser.add_argument('--dnn_weights', type=str, required=True, help='Path to the DNN model weights file.')
    parser.add_argument('--test_data', type=str, required=True, help='Path to the test data file (pickle format).')
    args = parser.parse_args()

    # 모델 가중치 경로 설정
    LSTM_WEIGHTS_PATH = args.lstm_weights
    DNN_WEIGHTS_PATH = args.dnn_weights
    # 테스트 데이터 경로 설정
    TEST_DATA_PATH = args.test_data

    # 모델 로드
    lstm_model = load_lstm_model(LSTM_WEIGHTS_PATH)
    dnn_model = load_dnn_model(DNN_WEIGHTS_PATH)

    # 테스트 데이터 로드 (12번째 샘플: 10월 17일 11시 예측용 데이터)
    with open(TEST_DATA_PATH, 'rb') as f:
        test_X = pickle.load(f)
    
    for i in range(24):
      sample_index = i
      sample_lstm_input = test_X[sample_index : sample_index + 1]
      print(f"\n테스트용 LSTM 입력 데이터 형태: {sample_lstm_input.shape}")

      # [추가됨] 기존 TOD 신호 계획 정의
      tod_plan = {
          (0, 5): [24, 31, 20, 17, 17, 17],
          (6, 6): [21, 36, 23, 19, 17, 20],
          (7, 9): [21, 41, 31, 23, 17, 21],
          (10, 13): [20, 41, 31, 23, 20, 21],
          (14, 21): [31, 31, 23, 26, 19, 26],
          (22, 23): [24, 31, 20, 17, 17, 17]
      }
      def get_tod_signal_for_hour(hour, plan):
          for (start, end), signal in plan.items():
              if start <= hour <= end:
                  return np.array(signal)
          return None

      # 프레임워크 인스턴스 생성
      framework = Signal_Framework(lstm_model, dnn_model)

      # 최적화 실행
      best_signal, best_speed = framework.optimize_signal_times(
          lstm_input=sample_lstm_input,
          population_size=100,
          num_generations=200,
          num_parents_mating=10,
          mutation_rate=0.3
      )

      # [추가됨] 기존 TOD 신호와 성능 비교
      hour_of_day = sample_index % 24
      tod_signal = get_tod_signal_for_hour(hour_of_day, tod_plan)
      
      if tod_signal is not None:
          print("\n--- 기존 TOD 신호 성능 평가 ---")
          tod_speed = framework.evaluate_signal(tod_signal)
          print(f"10월 17일 {hour_of_day:02d}시의 TOD 신호: {tod_signal}")
          print(f"기존 TOD 신호의 예상 속도: {tod_speed:.4f}")
      else:
          print(f"\n{hour_of_day}시에 해당하는 TOD 계획을 찾을 수 없습니다.")

      # 최종 결과 출력
      print("\n--- GA 최적화 최종 결과 ---")
      print(f"최적의 신호 시간 조합: {best_signal}")
      print(f"예상되는 최고 가중 평균 속도: {best_speed:.4f}")
      
      # [추가됨] 성능 비교 요약
      if tod_signal is not None:
          improvement = ((best_speed - tod_speed) / tod_speed) * 100 if tod_speed > 0 else float('inf')
          print("\n--- 성능 비교 요약 ---")
          print(f"예상 개선율: {improvement:.2f}%")

      # 수렴 그래프 시각화
      framework.plot_fitness_trend()
    
