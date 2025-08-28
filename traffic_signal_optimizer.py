import argparse
import os
import pickle
import matplotlib as mat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import trange, tqdm

# For running on Google Colab
# from google.colab import drive
# drive.mount('/content/drive')

def set_publication_style():
  mat.rc('font', family='DejaVu Serif')
  mat.rcParams['mathtext.fontset'] = 'dejavuserif'
  mat.rcParams['font.size'] = 16
  mat.rcParams['axes.labelsize'] = 16
  mat.rcParams['xtick.labelsize'] = 14
  mat.rcParams['ytick.labelsize'] = 14
  mat.rcParams['legend.fontsize'] = 14
  # mat.rcParams['axes.titlesize'] = 12
  mat.rcParams['savefig.dpi'] = 1200       # Line Art DPI
  mat.rcParams['figure.dpi'] = 1200        # DPI
  mat.rcParams['lines.linewidth'] = 1.5
  mat.rcParams['axes.grid'] = True
  mat.rcParams['axes.unicode_minus'] = False
  mat.rcParams['axes.xmargin'] = 0.1
  mat.rcParams['axes.ymargin'] = 0.1

class SignalOptimizer:
    """
    DNN models and genetic algorithm to find the optimal signal combination.
    Aims to minimize the MSE between predicted speeds and tragets speeds.
    6 signal phases with defined min and max times.
    1. Initialize with DNN model, scaler, base features, and actual speeds.
    2. Prepare DNN inputs for all directions.
    3. Evaluate fitness of population based on negative MSE.
    4. Select parents, perform crossover and mutation to create new population.
    5. Optimize over specified generations and return best signal combination.
    6. Plot fitness trend over generations.
    7. Used for both single timestep and daily optimizations.
    8. Also used for sensitivity analysis by varying one signal at a time.
    """
    def __init__(self, dnn_model, scaler, base_input_features, actual_speeds):
        self.dnn_model = dnn_model
        self.scaler = scaler
        self.base_features = base_input_features
        self.actual_speeds = actual_speeds
        self.min_signal_times = np.array([10, 31, 15, 11, 8, 15])
        self.max_signal_times = np.array([90, 90, 50, 50, 50, 80])
        self.best_fitness_per_generation = []

    def _prepare_dnn_inputs(self, signal_combination):
        dnn_inputs = np.zeros((6, 24))
        for i in range(6):
            temp_features = np.concatenate([self.base_features, signal_combination, [i]])
            dnn_inputs[i, :] = temp_features
        return dnn_inputs

    def evaluate_fitness(self, population):
        fitness_scores = np.zeros(len(population))
        for i, individual_signal in enumerate(population):
            if np.sum(individual_signal) > 200:
                fitness_scores[i] = -1e9 
                continue
            dnn_inputs = self._prepare_dnn_inputs(individual_signal)
            dnn_inputs_scaled = self.scaler.transform(dnn_inputs)
            predicted_speeds = self.dnn_model.predict(dnn_inputs_scaled, verbose=0).flatten()
            mse = np.mean((self.actual_speeds - predicted_speeds)**2)
            fitness_scores[i] = -mse
        return fitness_scores

    def _select_parents(self, population, fitness, num_parents):
        parents = np.empty((num_parents, population.shape[1]))
        sorted_indices = np.argsort(fitness)[::-1]
        for i in range(num_parents):
            parents[i, :] = population[sorted_indices[i], :]
        return parents

    def _crossover(self, parents, offspring_size):
        offspring = np.empty(offspring_size)
        crossover_point = offspring_size[1] // 2
        for k in range(offspring_size[0]):
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring

    def _mutate(self, offspring, mutation_rate):
        for idx in range(offspring.shape[0]):
            if np.random.rand() < mutation_rate:
                gene_idx = np.random.randint(0, offspring.shape[1])
                random_value = np.random.randint(-5, 6)
                mutated_gene = offspring[idx, gene_idx] + random_value
                offspring[idx, gene_idx] = np.clip(
                    mutated_gene, self.min_signal_times[gene_idx], self.max_signal_times[gene_idx]
                )
        return offspring

    def _create_initial_population(self, population_size):
        population = np.zeros((population_size, len(self.min_signal_times)))
        for i in range(population_size):
            for j in range(len(self.min_signal_times)):
                population[i, j] = np.random.randint(
                    self.min_signal_times[j], self.max_signal_times[j] + 1
                )
        return population

    def optimize(self, population_size=100, num_generations=200, num_parents=10, mutation_rate=0.3):
        population = self._create_initial_population(population_size)
        print("Starting Genetic Algorithm optimization (Objective: Minimize MSE)...")
        for generation in trange(num_generations, desc="Optimizing", leave=False):
            fitness = self.evaluate_fitness(population)
            self.best_fitness_per_generation.append(np.max(fitness))
            parents = self._select_parents(population, fitness, num_parents)
            offspring_size = (population_size - parents.shape[0], population.shape[1])
            offspring_crossover = self._crossover(parents, offspring_size)
            offspring_mutation = self._mutate(offspring_crossover, mutation_rate)
            population[0:parents.shape[0], :] = parents
            population[parents.shape[0]:, :] = offspring_mutation
        final_fitness = self.evaluate_fitness(population)
        best_match_idx = np.argmax(final_fitness)
        best_solution = population[best_match_idx, :]
        best_fitness = final_fitness[best_match_idx]
        return best_solution.astype(int), best_fitness
    
    def plot_fitness_trend(self, save_path):
        set_publication_style()
        plt.figure(figsize=(8, 5))
        plt.plot(self.best_fitness_per_generation)
        # plt.title('GA Fitness Convergence (Objective: Minimize MSE)')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness (-MSE)')
        plt.ylim(top=0)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, format='eps', bbox_inches='tight')
        print(f"✅ Fitness trend graph saved to '{save_path}'")
        plt.close()

def run_single_timestep_optimization(dnn_model, scaler, test_X, test_Y, day_offset, hour, args):
    closed_loop = args.closed_loop
    start_row = (day_offset * 24 * 6) + (hour * 6)
    end_row = start_row + 6
    print(f"\nOptimizing for Day: {day_offset}, Hour: {hour} (Rows {start_row} to {end_row-1})")
    base_features = test_X.iloc[start_row, :17].values
    original_signals = test_X.iloc[start_row, 17:23].values.astype(int)
    actual_speeds = test_Y.iloc[start_row : end_row].values.flatten()
    if closed_loop == 1:
      # Use fixed example data for closed loop testing
      base_features = np.array([179.05685, 222.49242, 310.61502, 192.78485, 128.89693, 94.2162, 274.18042, 57.862648, 201.37315, 130.39098, 20.33832,
                                250.28032, 77.827484, 152.72862,  197.37221, 63.327084, 0], dtype=np.int64)
      original_signals = np.array([24, 31, 20, 17, 17, 16], dtype=np.int64)
      actual_speeds = np.array([21.18259117, 27.09952524, 19.49596127, 25.47332749,	28.10950259, 13.14489088], dtype=np.float64)

    print(f"base_features: {type(base_features)}, original_signals: {type(original_signals)}, actual_speeds: {type(actual_speeds)}")
    print(f"base_features: {base_features}, original_signals: {original_signals}, actual_speeds: {actual_speeds}")
    print(f"base_features: {base_features.shape}, original_signals: {original_signals.shape}, actual_speeds: {actual_speeds.shape}")
    print(f"base_features: {base_features.ndim}, original_signals: {original_signals.ndim}, actual_speeds: {actual_speeds.ndim}")
    print(f"base_features: {base_features.dtype}, original_signals: {original_signals.dtype}, actual_speeds: {actual_speeds.dtype}")  

    optimizer = SignalOptimizer(dnn_model, scaler, base_features, actual_speeds)
    best_signal, best_fitness_score = optimizer.optimize(
        population_size=args.pop_size, num_generations=args.gens,
        num_parents=args.num_parents, mutation_rate=args.mut_rate
    )
    optimal_dnn_inputs = optimizer._prepare_dnn_inputs(best_signal)
    optimal_dnn_inputs_scaled = scaler.transform(optimal_dnn_inputs)
    predicted_speeds_with_optimal_signal = dnn_model.predict(optimal_dnn_inputs_scaled, verbose=0).flatten()
    actual_avg_speed = np.mean(actual_speeds)
    predicted_avg_speed = np.mean(predicted_speeds_with_optimal_signal)
    print("\n" + "="*60)
    print("Optimization Results (Objective: Minimize MSE)")
    print("="*60)
    print(f"🔹 Original Signals: {original_signals}")
    print(f"🔹 Actual Average Speed: {actual_avg_speed:.2f} km/h")
    print("-" * 60)
    print(f"✅ Optimal Signals (to match reality): {best_signal}")
    print(f"✅ Minimized MSE: {-best_fitness_score:.4f}")
    print(f"   (Predicted Average Speed with this signal: {predicted_avg_speed:.2f} km/h)")
    print("-" * 60)
    improvement = ((predicted_avg_speed - actual_avg_speed) / actual_avg_speed) * 100 if actual_avg_speed > 0 else float('inf')
    print(f"🚀 Potential Improvement: {improvement:.2f}%")
    print("="*60)
    print("\n" + "="*60)
    print("Detailed Speed Comparison per Direction")
    print("="*60)
    comparison_df = pd.DataFrame({
        'Direction': range(6), 'Actual_Speed': actual_speeds,
        'Predicted_Speed_with_Optimal_Signal': predicted_speeds_with_optimal_signal
    })
    print(comparison_df.round(2))
    print("="*60)
    fitness_plot_path = f'fitness_trend_day_{day_offset}_hour_{hour}.eps'
    optimizer.plot_fitness_trend(save_path=fitness_plot_path)

def run_daily_optimization(dnn_model, scaler, test_X, test_Y, day_offset, args):
    print(f"\nStarting optimization for day {day_offset} (24 timesteps).")
    for hour in range(24):
        run_single_timestep_optimization(dnn_model, scaler, test_X, test_Y, day_offset, hour, args)

def run_single_timestep_sensitivity_analysis(dnn_model, scaler, test_X, test_Y, day_offset, hour):
    start_row = (day_offset * 24 * 6) + (hour * 6)
    end_row = start_row + 6
    print(f"\nRunning Sensitivity Analysis for Day: {day_offset}, Hour: {hour}")
    base_features = test_X.iloc[start_row, :17].values
    original_signals = test_X.iloc[start_row, 17:23].values.astype(int)
    actual_speeds = test_Y.iloc[start_row : end_row].values.flatten()
    optimizer_for_bounds = SignalOptimizer(dnn_model, scaler, base_features, actual_speeds)
    min_times, max_times = optimizer_for_bounds.min_signal_times, optimizer_for_bounds.max_signal_times
    
    num_signals = 6
    set_publication_style()
    fig_speed, axes_speed = plt.subplots(2, 3, figsize=(20, 10))
    # fig_speed.suptitle(f'Speed Sensitivity Analysis for Day {day_offset}, Hour {hour}', fontsize=16)
    axes_speed = axes_speed.flatten()
    
    fig_fitness, axes_fitness = plt.subplots(2, 3, figsize=(20, 10))
    # fig_fitness.suptitle(f'Fitness (-MSE) Sensitivity Analysis for Day {day_offset}, Hour {hour}', fontsize=16)
    axes_fitness = axes_fitness.flatten()

    for i in range(num_signals):
        signal_times_to_test = np.arange(min_times[i], max_times[i] + 1)
        speed_results = []
        fitness_results = [] 
        for time in signal_times_to_test:
            temp_signals = original_signals.copy()
            temp_signals[i] = time
            dnn_inputs = np.zeros((6, 24))
            for j in range(6):
                dnn_inputs[j, :] = np.concatenate([base_features, temp_signals, [j]])
            dnn_inputs_scaled = scaler.transform(dnn_inputs)
            preds = dnn_model.predict(dnn_inputs_scaled, verbose=0).flatten()
            speed_results.append(np.mean(preds))
            fitness_results.append(-np.mean((actual_speeds - preds)**2)) 

        # Speed Graph
        ax_speed = axes_speed[i]
        ax_speed.plot(signal_times_to_test, speed_results, marker='o', linestyle='-')
        ax_speed.set_title(f'Signal Phase {i}')
        ax_speed.set_xlabel('Signal Time (s)')
        ax_speed.set_ylabel('Avg. Predicted Speed (kph)')
        ax_speed.grid(True)
        
        # Plot Fitness (-MSE) Graph
        ax_fitness = axes_fitness[i]
        ax_fitness.plot(signal_times_to_test, fitness_results, marker='o', linestyle='-', color='green')
        ax_fitness.set_title(f'Signal Phase {i}')
        ax_fitness.set_xlabel('Signal Time (s)')
        ax_fitness.set_ylabel('Fitness (-MSE)')
        ax_fitness.grid(True)

    fig_speed.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path_speed = f'speed_sensitivity_analysis_day_{day_offset}_hour_{hour}.eps'
    fig_speed.savefig(save_path_speed, bbox_inches='tight')
    print(f"✅ Speed sensitivity analysis plot saved to '{save_path_speed}'")
    plt.close(fig_speed)
    
    fig_fitness.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path_fitness = f'fitness_sensitivity_analysis_day_{day_offset}_hour_{hour}.eps'
    fig_fitness.savefig(save_path_fitness, bbox_inches='tight')
    print(f"✅ Fitness (-MSE) sensitivity analysis plot saved to '{save_path_fitness}'")
    plt.close(fig_fitness)

def run_daily_sensitivity_analysis(dnn_model, scaler, test_X, test_Y, day_offset):
    print(f"\nRunning Average Sensitivity Analysis for Day {day_offset}...")
    num_signals = 6
    day_start_row = day_offset * 24 * 6
    dummy_base_features = test_X.iloc[0, :17].values
    optimizer_for_bounds = SignalOptimizer(dnn_model, scaler, dummy_base_features, None)
    min_times, max_times = optimizer_for_bounds.min_signal_times, optimizer_for_bounds.max_signal_times
    
    all_day_speeds = [[] for _ in range(num_signals)]
    all_day_fitness = [[] for _ in range(num_signals)] 

    for hour in tqdm(range(24), desc=f"Analyzing Day {day_offset}"):
        start_row = day_start_row + (hour * 6)
        end_row = start_row + 6
        base_features = test_X.iloc[start_row, :17].values
        original_signals = test_X.iloc[start_row, 17:23].values.astype(int)
        actual_speeds = test_Y.iloc[start_row : end_row].values.flatten()
        
        for i in range(num_signals):
            signal_times_to_test = np.arange(min_times[i], max_times[i] + 1)
            speeds_for_this_phase = []
            fitness_for_this_phase = [] 
            for time in signal_times_to_test:
                temp_signals = original_signals.copy()
                temp_signals[i] = time
                dnn_inputs = np.zeros((6, 24))
                for j in range(6):
                    dnn_inputs[j, :] = np.concatenate([base_features, temp_signals, [j]])
                dnn_inputs_scaled = scaler.transform(dnn_inputs)
                preds = dnn_model.predict(dnn_inputs_scaled, verbose=0).flatten()
                speeds_for_this_phase.append(np.mean(preds))
                fitness_for_this_phase.append(-np.mean((actual_speeds - preds)**2)) 
            all_day_speeds[i].append(speeds_for_this_phase)
            all_day_fitness[i].append(fitness_for_this_phase) 

    # Speed Graph
    set_publication_style()
    fig_speed, axes_speed = plt.subplots(2, 3, figsize=(20, 10))
    # fig_speed.suptitle(f'Average Speed Sensitivity Analysis for Day {day_offset}', fontsize=16)
    axes_speed = axes_speed.flatten()
    for i in range(num_signals):
        signal_times_to_test = np.arange(min_times[i], max_times[i] + 1)
        max_len = max(len(arr) for arr in all_day_speeds[i])
        padded_speeds = [np.pad(arr, (0, max_len - len(arr)), 'edge') for arr in all_day_speeds[i]]
        average_speeds = np.mean(padded_speeds, axis=0)
        ax = axes_speed[i]
        ax.plot(signal_times_to_test, average_speeds, marker='o', linestyle='-')
        ax.set_title(f'Signal Phase {i}')
        ax.set_xlabel('Signal Time (seconds)')
        ax.set_ylabel('Avg. Predicted Speed (km/h)')
        ax.grid(True)
    fig_speed.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path_speed = f'avg_speed_sensitivity_day_{day_offset}.eps'
    fig_speed.savefig(save_path_speed, bbox_inches='tight')
    print(f"✅ Average speed sensitivity plot saved to '{save_path_speed}'")
    plt.close(fig_speed)

    # Fitness (-MSE) Graph
    fig_fitness, axes_fitness = plt.subplots(2, 3, figsize=(20, 10))
    # fig_fitness.suptitle(f'Average Fitness (-MSE) Sensitivity Analysis for Day {day_offset}', fontsize=16)
    axes_fitness = axes_fitness.flatten()
    for i in range(num_signals):
        signal_times_to_test = np.arange(min_times[i], max_times[i] + 1)
        max_len = max(len(arr) for arr in all_day_fitness[i])
        padded_fitness = [np.pad(arr, (0, max_len - len(arr)), 'edge') for arr in all_day_fitness[i]]
        average_fitness = np.mean(padded_fitness, axis=0)
        ax = axes_fitness[i]
        ax.plot(signal_times_to_test, average_fitness, marker='o', linestyle='-', color='green')
        ax.set_title(f'Signal Phase {i}')
        ax.set_xlabel('Signal Time (seconds)')
        ax.set_ylabel('Average Fitness (-MSE)')
        ax.grid(True)
    fig_fitness.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path_fitness = f'avg_fitness_sensitivity_day_{day_offset}.eps'
    fig_fitness.savefig(save_path_fitness, bbox_inches='tight')
    print(f"✅ Average Fitness (-MSE) sensitivity plot saved to '{save_path_fitness}'")
    plt.close(fig_fitness)


def main(args):
    print("Loading model, scaler, and data...")
    if not all(os.path.exists(p) for p in [args.model_path, args.scaler_path, args.data_path, args.label_path]):
        print("❌ Error: One or more file paths are invalid. Please check the paths.")
        return
    dnn_model = load_model(args.model_path)
    with open(args.scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(args.data_path, 'rb') as f:
        test_X = pickle.load(f)
    with open(args.label_path, 'rb') as f:
        test_Y = pickle.load(f)

    total_days_available = len(test_X) // (24 * 6)
    if args.day_offset >= total_days_available:
        print(f"❌ Error: day_offset {args.day_offset} is out of bounds. Dataset has {total_days_available} days (0 to {total_days_available - 1}).")
        return
    if not (0 <= args.hour < 24):
        print(f"❌ Error: hour {args.hour} is invalid. Must be between 0 and 23.")
        return

    if args.mode == 'optimize_single':
        run_single_timestep_optimization(dnn_model, scaler, test_X, test_Y, args.day_offset, args.hour, args)
    elif args.mode == 'optimize_daily':
        run_daily_optimization(dnn_model, scaler, test_X, test_Y, args.day_offset, args)
    elif args.mode == 'analyze_single':
        run_single_timestep_sensitivity_analysis(dnn_model, scaler, test_X, test_Y, args.day_offset, args.hour)
    elif args.mode == 'analyze_daily':
        run_daily_sensitivity_analysis(dnn_model, scaler, test_X, test_Y, args.day_offset)
    else:
        print(f"❌ Error: Invalid mode '{args.mode}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize or analyze traffic signals using a DNN model.')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['optimize_single', 'optimize_daily', 'analyze_single', 'analyze_daily'],
                        help='Execution mode.')
    parser.add_argument('--closed_loop', type=int, default=0, help='Setting closed loop (0=Normal loop, 1=Closed loop).')
    parser.add_argument('--day_offset', type=int, default=0, help='Which day to use from the start of the dataset (0=first day).')
    parser.add_argument('--hour', type=int, default=0, help='Which hour to use (0-23). Only for "single" modes.')
    
    DRIVE_BASE_PATH = '/content/drive/MyDrive/Colab_Project'
    parser.add_argument('--model_path', type=str, 
                        default=os.path.join(DRIVE_BASE_PATH, 'checkpoint/model_dnn_lr0.001_seed42_best.keras'), 
                        help='Path to the trained Keras DNN model file.')
    parser.add_argument('--scaler_path', type=str, 
                        default=os.path.join(DRIVE_BASE_PATH, 'checkpoint/scaler_dnn_lr0.001_seed42.pkl'), 
                        help='Path to the trained scaler file.')
    parser.add_argument('--data_path', type=str, 
                        default=os.path.join(DRIVE_BASE_PATH, 'datasets/test_X.pkl'), 
                        help='Path to the test data (test_X.pkl).')
    parser.add_argument('--label_path', type=str, 
                        default=os.path.join(DRIVE_BASE_PATH, 'datasets/test_Y.pkl'), 
                        help='Path to the test labels (test_Y.pkl).')
    
    parser.add_argument('--pop_size', type=int, default=100, help='Population size for GA.')
    parser.add_argument('--gens', type=int, default=200, help='Number of generations for GA.')
    parser.add_argument('--num_parents', type=int, default=10, help='Number of parents for mating.')
    parser.add_argument('--mut_rate', type=float, default=0.3, help='Mutation rate for GA.')
    
    args = parser.parse_args()
    main(args)
