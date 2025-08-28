import pickle
import numpy as np
import os

class TrafficDataLoader:
    """
    A class that loads traffic data (.pkl) and presents it as a NumPy array.

    Usage:
        data_loader = TrafficDataLoader(data_path='./data/traffic_volume')
        train_X = data_loader.train_X
        train_Y = data_loader.train_Y
    """
    def __init__(self, data_path='./data/traffic_volume'):
        """
        Initializes the TrafficDataLoader with the path to the data directory.
        
        Args:
            data_path (str): Path to the directory containing the .pkl files.
        """
        self.data_path = data_path
        self._load_all_data()
        print("✅ Data loaded and prepared successfully!")

    def _load_pickle(self, file_name):
        """ Helper function to load a pickle file from the data path."""
        file_path = os.path.join(self.data_path, file_name)
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"❌ Error: '{file_path}' file not found. Please check the file path.")
            return None

    def _load_all_data(self):
        """Load all training, validation, and test data and store them as instance variables."""
        # Load all data files
        train_X_raw = self._load_pickle('train_X.pkl')
        self.train_Y = self._load_pickle('train_Y.pkl')
        val_X_raw = self._load_pickle('val_X.pkl')
        self.val_Y = self._load_pickle('val_Y.pkl')
        test_X_raw = self._load_pickle('test_X.pkl')
        self.test_Y = self._load_pickle('test_Y.pkl')

        # Convert to NumPy array (keeps data types consistent and creates a copy)
        self.train_X = np.array(train_X_raw)
        self.val_X = np.array(val_X_raw)
        self.test_X = np.array(test_X_raw)

# Ensure that the data is loaded correctly
if __name__ == '__main__':
    # 1. Create an instance of the TrafficDataLoader
    print("Loading traffic data...")
    loader = TrafficDataLoader()

    # 2. Print the shapes of the loaded data
    print("\n--- Data Shapes ---")
    print(f"Train X shape: {loader.train_X.shape}")
    print(f"Train Y shape: {loader.train_Y.shape}")
    print(f"Validation X shape: {loader.val_X.shape}")
    print(f"Validation Y shape: {loader.val_Y.shape}")
    print(f"Test X shape: {loader.test_X.shape}")
    print(f"Test Y shape: {loader.test_Y.shape}")

    # 3. Print a sample of the data
    print("\n--- Data Samples ---")
    print("Train X sample:\n", loader.train_X[0, :2, :])
    print("Train Y sample:\n", loader.train_Y[0, :])