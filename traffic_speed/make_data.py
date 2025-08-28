import pandas as pd
import numpy as np

# 1. Load the CSV file
df = pd.read_csv('./data/traffic_speed/dnn_train.csv')

# 2. Define feature and target columns
# Assuming the CSV has columns named 'D0' to 'D15', 'holiday', 'phase0' to 'phase5', 'approach', and 'speed'
# Adjust the column names based on the actual CSV structure
feature_cols = [
    'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 
    'D12', 'D13', 'D14', 'D15', 'holiday', 'phase0', 'phase1', 'phase2', 'phase3',
    'phase4', 'phase5', 'approach'
]
target_col = ['speed']

X = df[feature_cols]
Y = df[target_col]

n_samples = len(df)
print(f"Total samples: {n_samples}")
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# 4. Split the data sequentially
# data 0501 - 0930
X_train = X.iloc[:22031+1]
Y_train = Y.iloc[:22031+1]

# data 1001 - 1015
X_val = X.iloc[22031+1:24191+1]
Y_val = Y.iloc[22031+1:24191+1]

# data 1016 - 1031
X_test = X.iloc[24191+1:]
Y_test = Y.iloc[24191+1:]

# 5. Save the split data as pickle files _X.pkl and _Y.pkl
X_train.to_pickle('./data/traffic_speed/train_X.pkl')
Y_train.to_pickle('./data/traffic_speed/train_Y.pkl')
X_val.to_pickle('./data/traffic_speed/val_X.pkl')
Y_val.to_pickle('./data/traffic_speed/val_Y.pkl')
X_test.to_pickle('./data/traffic_speed/test_X.pkl')
Y_test.to_pickle('./data/traffic_speed/test_Y.pkl')

# 6. Print confirmation and shapes of the split data
print("✅ Data split and saved as pickle files.")
print(f'X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}')
print(f'X_val shape: {X_val.shape},   Y_val shape: {Y_val.shape}')
print(f'X_test shape: {X_test.shape},  Y_test shape: {Y_test.shape}')