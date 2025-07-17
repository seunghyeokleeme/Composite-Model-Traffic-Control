import argparse
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from datasets.speed_dataset import SpeedDataLoader

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)

def plot_loss_history(history, filename, result_dir):
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

def create_dnn_model(num_hidden_layers = 15, num_dense_node = 64, leaky_relu = 0.02):
    model = Sequential()
    model.add(Dense(num_dense_node, input_shape=(24,)))
    model.add(LeakyReLU(alpha=leaky_relu))

    for _ in range(num_hidden_layers):
        model.add(Dense(num_dense_node))
        model.add(LeakyReLU(alpha=leaky_relu))

    model.add(Dense(1, activation='linear'))
    return model

def main():
    parser = argparse.ArgumentParser(description='Train traffic prediction models for comparative experiments.')
    parser.add_argument('--model_type', default='simple_attention', type=str, choices=['dnn', 'simple_attention', 'deep_attention'], help='Model architecture to train.')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate.')
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
    train_X, train_Y, val_X, val_Y = data_loader.train_X, data_loader.train_Y, data_loader.val_X, data_loader.val_Y

    if args.model_type == 'dnn':
        model = create_dnn_model()
    
    model.summary()
    model.compile(optimizer=Adam(learning_rate=args.lr), loss='mse')

    run_name = f"{model.name}_lr{args.lr}_seed{args.seed}"
    checkpoint_path = os.path.join(args.ckpt_dir, f"{run_name}_best.keras")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=50, verbose=1),
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    ]

    print(f"\n--- Training {run_name} ---")
    history = model.fit(
        train_X, train_Y, validation_data=(val_X, val_Y),
        epochs=args.num_epoch, batch_size=args.batch_size,
        callbacks=callbacks, shuffle=False
    )

    print(f"\n✅ Training finished for {run_name}.")
    print(f"📈 Minimum validation loss: {min(history.history['val_loss']):.4f}")
    plot_loss_history(history, run_name, args.result_dir)

if __name__ == '__main__':
    main()