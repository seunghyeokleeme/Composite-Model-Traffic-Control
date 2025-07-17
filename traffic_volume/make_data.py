import pandas as pd
import numpy as np
import pickle

# --- Config ---
INPUT_CSV_PATH = './data/traffic_volume/교통량_방면별.csv'
WINDOW_SIZE = 24
HORIZON = 1
OUTPUT_X_FILENAME = 'data_X.pkl'
OUTPUT_Y_FILENAME = 'data_Y.pkl'

# --- Helper Function ---
def create_sliding_window_data(data, window_size, horizon):
    """Function to generate time series data (X, Y) using sliding window technique"""
    X, Y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        # Use data from current position (i) to window_size as input (X)
        X.append(data[i:(i + window_size)])
        # Immediately after that, use the data as much as horizon as the correct answer (y).
        Y.append(data[i + window_size:(i + window_size + horizon)])
    return np.array(X), np.array(Y)

try:
    # 1. Loading and preprocessing the original data
    df_long = pd.read_csv(INPUT_CSV_PATH)
    print("Complete data loaded successfully.")

    df_long['date'] = pd.to_datetime(df_long['date'])
    df_long.sort_values(by='date', inplace=True)

    df_pivot = df_long.pivot_table(
        index='date', 
        columns='방면명', 
        values=['직진', '좌회전', '우회전']
    )
    df_pivot.columns = [f'{direction.replace("→", "_")}_{turn}' for turn, direction in df_pivot.columns]
    
    df_holiday = df_long[['date', '빨간날']].drop_duplicates().set_index('date')
    df_wide = pd.merge(df_pivot, df_holiday, on='date', how='left')
    df_wide.fillna(0, inplace=True)
    print("Data pivoted and merged successfully.")

    # 2. Select the 17 features requested by the user in the correct order.
    feature_columns = [
        '교대_연산교차로_직진', '교대_연산교차로_우회전',
        '부산시청_연산교차로_직진', '부산시청_연산교차로_좌회전', '부산시청_연산교차로_우회전',
        '사직운동장_연산교차로_직진', '사직운동장_연산교차로_좌회전', '사직운동장_연산교차로_우회전',
        '신리_연산교차로_직진', '신리_연산교차로_좌회전', '신리_연산교차로_우회전',
        '안락_연산교차로_직진', '안락_연산교차로_우회전',
        '연산터널_연산교차로_직진', '연산터널_연산교차로_좌회전', '연산터널_연산교차로_우회전',
        '빨간날'
    ]
    df_selected = df_wide[feature_columns]
    print("Selected features successfully.")

    # 3. Create sliding window data
    feature_data = df_selected.values
    X_data, Y_data_full = create_sliding_window_data(feature_data, WINDOW_SIZE, HORIZON)
    
    # 4. Adjust the shape of X_data and Y_data
    Y_data = Y_data_full[:, :, :16]
    
    # If the last dimension of Y_data is 1, we can squeeze it to remove the singleton dimension.
    Y_data = Y_data.squeeze(axis=1)

    # 5. Print the final shapes of X_data and Y_data
    print(f"X_data shape: {X_data.shape}")
    print(f"Y_data shape: {Y_data.shape}")

    # 6. Save the processed data to pickle files
    with open(OUTPUT_X_FILENAME, 'wb') as f:
        pickle.dump(X_data, f)
    print(f"Success: Processed data saved to '{OUTPUT_X_FILENAME}'.")

    with open(OUTPUT_Y_FILENAME, 'wb') as f:
        pickle.dump(Y_data, f)
    print(f"Success: Processed data saved to '{OUTPUT_Y_FILENAME}'.")

except Exception as e:
    print(f"An error occurred: {e}")
    exit()