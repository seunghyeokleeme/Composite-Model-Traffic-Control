import numpy as np
import pickle

# --- Config ---
INPUT_FILENAME = './data/traffic_volume/data_Y.pkl'
OUTPUT_FILENAME = './data/traffic_volume/test_Y.pkl'
# Define the slice indices
SLICE_START_INDEX = 0
SLICE_END_INDEX = 42

try:
    with open(INPUT_FILENAME, 'rb') as f:
        original_data = pickle.load(f)
    print(f"Original data loaded successfully from '{INPUT_FILENAME}'")
except FileNotFoundError:
    print(f"Error: '{INPUT_FILENAME}' file not found. Please check the file path.")
    exit()

# 1. Check the shape of the original data
sliced_data = original_data[SLICE_START_INDEX:SLICE_END_INDEX, : ]

# 2. Print the shape of the sliced data
print(f"sliced_data shape: {sliced_data.shape}")

# 4. Save the sliced data to a new file
with open(OUTPUT_FILENAME, 'wb') as f:
    pickle.dump(sliced_data, f)

print(f"Success: Sliced data saved to '{OUTPUT_FILENAME}'")