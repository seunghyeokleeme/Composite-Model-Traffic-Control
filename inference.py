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
    
    # 저장된 가중치 로드
    model.load_weights(weights_path)
    print("LSTM 모델 로드 완료.")
    return model

def load_dnn_model(weights_path):
    """
    저장된 가중치를 사용하여 DNN 모델을 정의하고 불러옵니다.
    traffic_speed/train.py의 모델 구조와 동일해야 합니다.
    """
    # 입력층 정의
    model = Sequential()
    model.add(Dense(64, input_shape=(24,)))
    model.add(LeakyReLU(alpha=0.02))

    for _ in range(15):
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.02))

    model.add(Dense(1, activation='linear'))
    
    # 저장된 가중치 로드
    model.load_weights(weights_path)
    print("DNN 모델 로드 완료.")
    return model

# --- 2. 신호 최적화 프레임워크 클래스 ---

class Signal_Framework:
  def __init__(self, lstm_model, dnn_model):
    # 최소/최대 녹색시간 (주황불 4초 제외)
    self.min_signal_times = np.array([10, 31, 15, 11, 8, 15])
    self.max_signal_times = np.array([45-4, 55-4, 45-4, 40-4, 35-4, 40-4]) # [41, 51, 41, 36, 31, 36]

    self.lstm_model = lstm_model
    self.dnn_model = dnn_model

    self.lstm_input = None
    self.lstm_output = None

    self.best_solution = None
    self.best_fitness_per_generation = []

  def _prepare_dnn_input(self, signal_values):
      """단일 신호 주기에 대한 DNN 입력 데이터를 준비하는 내부 함수"""
      # lstm_input에서 마지막 시간의 공휴일 정보(17번째 특징)를 가져옵니다.
      holiday_info = self.lstm_input[0, -1, 16] 
      
      # 16개의 예측 교통량, 1개의 공휴일 정보, 6개의 신호 시간 값을 결합합니다.
      base_features = np.concatenate([
          self.lstm_output.flatten(), # (1, 16) -> (16,)
          [holiday_info], 
          signal_values
      ])
      
      # 6개 방향에 대해 데이터를 복제하고, 각 행에 방향 인덱스(0~5)를 추가합니다.
      dnn_input = np.zeros((6, 24))
      for i in range(6):
          dnn_input[i, :] = np.concatenate([base_features, [i]])
          
      return dnn_input

  def evaluate_population(self, population):
      """전체 집단을 평가하는 함수"""
      all_dnn_inputs = []
      valid_indices = [] # 유효한 개체(신호 주기 합 <= 200)의 인덱스

      for i, individual in enumerate(population):
          if np.sum(individual) <= 200: # 최대 신호 주기 시간 (초)
              prepared_data = self._prepare_dnn_input(individual)
              all_dnn_inputs.append(prepared_data)
              valid_indices.append(i)
      
      if not all_dnn_inputs:
          return np.zeros(len(population))

      # 모든 유효한 개체의 데이터를 하나의 큰 배치로 결합
      all_dnn_inputs_combined = np.vstack(all_dnn_inputs)

      # 한 번에 모든 데이터에 대한 DNN 예측 수행
      all_predicted_speeds = self.dnn_model.predict(all_dnn_inputs_combined, verbose=0)

      # 각 개체에 대한 평균 속도(적합도) 계산
      fitness_scores = np.zeros(len(population))
      predicted_volumes = self.lstm_output.flatten()
      
      # 가중치 계산 (6개 방면에 대한 교통량 합)
      weights = np.array([
          np.sum(predicted_volumes[0:2]),   # 교대
          np.sum(predicted_volumes[2:5]),   # 시청
          np.sum(predicted_volumes[5:8]),   # 사직
          np.sum(predicted_volumes[8:11]),  # 신리
          np.sum(predicted_volumes[11:13]), # 안락
          np.sum(predicted_volumes[13:16])  # 터널
      ])
      
      total_volume = np.sum(predicted_volumes)
      if total_volume == 0: total_volume = 1 # 0으로 나누는 것 방지

      start_idx = 0
      for i, original_idx in enumerate(valid_indices):
          # 6개 방향에 대한 예측 속도 추출
          predicted_speeds_for_individual = all_predicted_speeds[start_idx : start_idx + 6].flatten()
          
          # 가중 평균 속도 계산
          weighted_avg_speed = np.sum(predicted_speeds_for_individual * weights) / total_volume
          fitness_scores[original_idx] = weighted_avg_speed
          start_idx += 6
          
      return fitness_scores

  def select_parents(self, population, fitness, num_parents):
      """부모 선택 함수 (엘리티즘)"""
      parents = np.empty((num_parents, population.shape[1]), dtype=np.int32)
      # fitness 값을 복사하여 원본이 변경되지 않도록 함
      temp_fitness = np.copy(fitness)
      for parent_num in range(num_parents):
          max_fitness_idx = np.argmax(temp_fitness)
          parents[parent_num, :] = population[max_fitness_idx, :]
          temp_fitness[max_fitness_idx] = -999999 # 중복 선택 방지
      return parents

  def crossover(self, parents, offspring_size):
      """교차(교배) 함수 (단일점 교배)"""
      offspring = np.empty(offspring_size, dtype=np.int32)
      crossover_point = offspring_size[1] // 2
      for k in range(offspring_size[0]):
          parent1_idx = k % parents.shape[0]
          parent2_idx = (k + 1) % parents.shape[0]
          offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
          offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
      return offspring

  def mutate(self, offspring_crossover, mutation_rate):
      """변이 함수"""
      for idx in range(offspring_crossover.shape[0]):
          if np.random.rand() < mutation_rate:
              random_index = np.random.randint(0, offspring_crossover.shape[1])
              random_value = np.random.randint(-5, 6)
              
              mutated_gene = offspring_crossover[idx, random_index] + random_value
              # clip을 사용하여 min/max 범위를 벗어나지 않도록 함
              offspring_crossover[idx, random_index] = np.clip(
                  mutated_gene, 
                  self.min_signal_times[random_index], 
                  self.max_signal_times[random_index]
              )
      return offspring_crossover

  def create_initial_population(self, population_size):
      """초기 집단 생성 함수"""
      population = np.zeros((population_size, len(self.min_signal_times)), dtype=np.int32)
      for i in range(population_size):
          for j in range(len(self.min_signal_times)):
              population[i, j] = np.random.randint(self.min_signal_times[j], self.max_signal_times[j] + 1)
      return population

  def optimize_signal_times(self, lstm_input, population_size=100, num_generations=200, num_parents_mating=10, mutation_rate=0.3):
      """신호 주기 최적화 메인 함수"""
      # 1. LSTM으로 미래 교통량 예측
      self.lstm_input = lstm_input
      self.lstm_output = self.lstm_model.predict(self.lstm_input, verbose=0)

      # 2. 유전 알고리즘 실행
      self.best_fitness_per_generation = []
      population = self.create_initial_population(population_size)

      print("유전 알고리즘 최적화 시작...")
      for generation in trange(num_generations):
          # 적합도 평가
          fitness = self.evaluate_population(population)
          self.best_fitness_per_generation.append(np.max(fitness))

          # 부모 선택 (엘리티즘)
          parents = self.select_parents(population, fitness, num_parents_mating)
          
          # 자손 생성 (교배 및 변이)
          offspring_size = (population_size - parents.shape[0], population.shape[1])
          offspring_crossover = self.crossover(parents, offspring_size)
          offspring_mutation = self.mutate(offspring_crossover, mutation_rate)
          
          # 다음 세대 구성
          population[0:parents.shape[0], :] = parents
          population[parents.shape[0]:, :] = offspring_mutation

      # 최종 세대의 적합도를 다시 평가하여 최적해 결정
      final_fitness = self.evaluate_population(population)
      best_match_idx = np.argmax(final_fitness)
      self.best_solution = population[best_match_idx, :]
      best_solution_fitness = final_fitness[best_match_idx]
      
      print("최적화 완료.")
      return self.best_solution, best_solution_fitness

  def plot_fitness_trend(self):
      """세대별 최고 적합도 추세 그래프 생성"""
      plt.figure(figsize=(10, 6))
      plt.plot(self.best_fitness_per_generation)
      plt.title('Generation vs. Best Fitness Trend')
      plt.xlabel('Generation')
      plt.ylabel('Best Fitness (Weighted Average Speed)')
      plt.grid(True)
      
      # [수정됨] 그래프를 이미지 파일로 저장하는 코드 추가
      output_filename = 'fitness_trend.png'
      plt.savefig(output_filename, dpi=300) # dpi 옵션으로 고해상도 저장
      print(f"Convergence graph saved to '{output_filename}'")
      
      plt.show()

# --- 3. 메인 실행 블록 ---

if __name__ == '__main__':
    # 경고 메시지 끄기 (선택 사항)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')

    parser = argparse.ArgumentParser(description='Optimize traffic signal timings using LSTM and DNN models.')
    parser.add_argument('--lstm_weights', type=str, required=True, help='Path to the LSTM model weights file.')
    parser.add_argument('--dnn_weights', type=str, required=True, help='Path to the DNN model weights file.')
    parser.add_argument('--test_data', type=str, required=True, help='Path to the test data file (pickle format).')
    parser.add_argument('--select_index', type=int, default=0, help='Index of the test sample to use for optimization.')
    args = parser.parse_args()

    # 모델 가중치 경로 설정
    LSTM_WEIGHTS_PATH = args.lstm_weights
    DNN_WEIGHTS_PATH = args.dnn_weights
    
    # 테스트 데이터 경로 설정
    TEST_DATA_PATH = args.test_data

    select_index = args.select_index

    # 모델 로드
    lstm_model = load_lstm_model(LSTM_WEIGHTS_PATH)
    dnn_model = load_dnn_model(DNN_WEIGHTS_PATH)

    # 테스트 데이터 로드 (첫 번째 샘플만 사용)
    with open(TEST_DATA_PATH, 'rb') as f:
        test_X = pickle.load(f)
    
    # LSTM 입력 데이터 준비 (Batch 차원 추가)
    sample_lstm_input = test_X[select_index:select_index+1] # (1, 24, 17) 형태
    print(f"\n테스트용 LSTM 입력 데이터 형태: {sample_lstm_input.shape}")

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

    # 결과 출력
    print("\n--- 최종 결과 ---")
    print(f"최적의 신호 시간 조합: {best_signal}")
    print(f"예상되는 최고 가중 평균 속도: {best_speed:.4f}")

    # 수렴 그래프 시각화
    framework.plot_fitness_trend()
