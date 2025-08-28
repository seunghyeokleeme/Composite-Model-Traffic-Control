import argparse
import os
import random
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from speed_dataset import SpeedDataLoader

def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)

def plot_loss_history(history, filename, result_dir):
    """Save the training and validation loss curves."""
    plt.figure(figsize=(15, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss Curve for {filename}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(result_dir, f'loss_curve_{filename}.png')
    plt.savefig(save_path)
    print(f"✅ Loss curve saved to '{save_path}'")
    plt.close()

def create_dnn_model(input_dim, num_hidden_layers=3, num_dense_node=64, leaky_relu=0.02):
    """
    Create a Deep Neural Network (DNN) model for traffic speed prediction.
    Args:
        input_dim (int): Number of features in the input data.
        num_hidden_layers (int): Number of hidden layers in the DNN.
    """
    model = Sequential(name='DNN_Model')
    model.add(Dense(num_dense_node, input_dim=input_dim))
    model.add(LeakyReLU(alpha=leaky_relu))

    for _ in range(num_hidden_layers):
        model.add(Dense(num_dense_node))
        model.add(LeakyReLU(alpha=leaky_relu))

    model.add(Dense(1, activation='linear'))
    return model

def main():
    parser = argparse.ArgumentParser(description='Train traffic prediction models for comparative experiments.')
    parser.add_argument('--model_type', default='dnn', type=str, choices=['dnn'], help='Model architecture to train.')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate.')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size.')
    parser.add_argument('--num_epoch', default=300, type=int, help='Number of epochs.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--data_dir', default='./datasets', type=str, help='Dataset directory.')
    parser.add_argument('--ckpt_dir', default='./checkpoint', type=str, help='Checkpoint directory.')
    parser.add_argument('--result_dir', default='./results', type=str, help='Results directory.')
    args = parser.parse_args()

    set_seeds(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    
    data_loader = SpeedDataLoader(data_path=args.data_dir)
    train_X, train_Y = data_loader.train_X, data_loader.train_Y
    val_X, val_Y = data_loader.val_X, data_loader.val_Y
    test_X, test_Y = data_loader.test_X, data_loader.test_Y
    print(f"✅ Data loaded. Train: {train_X.shape}, Val: {val_X.shape}, Test: {test_X.shape}")

    # -------- Data Scaling --------
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    val_X_scaled = scaler.transform(val_X)
    test_X_scaled = scaler.transform(test_X)
    print("✅ Data scaling complete.")

    run_name = f"dnn_lr{args.lr}_seed{args.seed}"
    scaler_path = os.path.join(args.ckpt_dir, f'scaler_{run_name}.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved to '{scaler_path}'")
    
    
    # -------- Model Creation --------
    model = create_dnn_model(input_dim=train_X_scaled.shape[1])
    model.summary()
    model.compile(optimizer=Adam(learning_rate=args.lr), loss='mse')

    checkpoint_path = os.path.join(args.ckpt_dir, f"model_{run_name}_best.keras")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    ]

    print(f"\n--- Training {run_name} ---")
    history = model.fit(
        train_X_scaled, train_Y,
        validation_data=(val_X_scaled, val_Y),
        epochs=args.num_epoch,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=False
    )

    print(f"\n✅ Training finished for {run_name}.")
    min_val_loss = min(history.history['val_loss'])
    print(f"📈 Minimum validation loss during training: {min_val_loss:.4f}")
    plot_loss_history(history, run_name, args.result_dir)

    print(f"\n--- Evaluating the best model on the test set ---")
    test_loss = model.evaluate(test_X_scaled, test_Y, verbose=0)
    print(f"🚀 Final Test Loss (MSE): {test_loss:.4f}")

if __name__ == '__main__':
    main()