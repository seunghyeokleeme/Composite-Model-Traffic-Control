import argparse
import os
import pandas as pd
import pickle
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from traffic_dataset import TrafficDataLoader
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Dropout,Layer,GRU, BatchNormalization,RNN,SimpleRNN
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras import initializers
from attention_layer import AttentionLayer

parser = argparse.ArgumentParser(description='Train the traffic prediction model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr', default=1e-3, type=float, dest='lr')
parser.add_argument('--batch_size', default=128, type=int, dest='batch_size')
parser.add_argument('--num_epoch', default=300, type=int, dest='num_epoch')
parser.add_argument('--seed', default=42, type=int, dest='seed')

parser.add_argument('--data_dir', default='./datasets', type=str, dest='data_dir')
parser.add_argument('--ckpt_dir', default='./checkpoint', type=str, dest='ckpt_dir')
parser.add_argument('--result_dir', default='./results', type=str, dest='result_dir')

parser.add_argument('--model_type', default='dnn', type=str, dest='model_type')

args = parser.parse_args()

#parameters
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch
seed_value = args.seed

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
result_dir = args.result_dir

model_type = args.model_type

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds(seed_value)

# Define paths for datasets and models
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset path '{data_dir}' does not exist. Please check the path.")

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
print(f"Using device: {device}")

def plot_loss_history(history, model_name):
    plt.figure(figsize=(15, 6))
    
    # Plot the loss curves
    plt.plot(history.history['loss'][3:], label='Training Loss')
    plt.plot(history.history['val_loss'][3:], label='Validation Loss')
    
    plt.title(f'Loss Curve for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(os.path.join(result_dir, f'loss_curve_{model_name}.png'))
    print(f"Loss curve saved as 'loss_curve_{model_name}.png'")
    plt.show()


data_loader = TrafficDataLoader(data_path=data_dir)

# train mode
train_X, train_Y, val_X, val_Y = data_loader.train_X, data_loader.train_Y, data_loader.val_X, data_loader.val_Y

def create_complex_model(input_shape=(24, 17), output_dim=16):
    input_layer = Input(shape=input_shape, name="Input_Tensor")
    x = Bidirectional(LSTM(512, return_sequences=True, activation='tanh'), name="Bi-LSTM_1")(input_layer)
    x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), name="Bi-LSTM_2")(x)
    x = AttentionLayer(name="Temporal_Attention")(x)
    x = Dense(2048, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(output_dim, name="Output_Layer")(x)
    model = Model(inputs=input_layer, outputs=output_layer, name="Complex_Model")
    return model

def create_simple_model(input_shape=(24, 17), output_dim=16):
    input_layer = Input(shape=input_shape, name="Input_Tensor")
    x = Bidirectional(LSTM(512, return_sequences=True, activation='tanh'), name="Bi-LSTM_1")(input_layer)
    x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), name="Bi-LSTM_2")(x)
    context_vector = AttentionLayer(name="Temporal_Attention")(x)
    output_layer = Dense(output_dim, name="Output_Layer")(context_vector)
    model = Model(inputs=input_layer, outputs=output_layer, name="Simple_Model")
    return model

def create_dnn_model(input_shape=(24, 17), output_dim=16):
    input_layer = Input(shape=input_shape, name="Input_Tensor")

    x = Bidirectional(LSTM(1000, return_sequences=True, activation='tanh'))(input_layer)
    x = Bidirectional(LSTM(500, return_sequences=True, activation='tanh'))(x)
    x = Bidirectional(LSTM(100, activation='tanh'))(x)

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

    model = Model(inputs=input_layer, outputs=output_layer, name="DNN_Model")
    return model


if model_type == 'complex':
    print("train complex model")
    model_a = create_complex_model()
    model_a.compile(optimizer=Adam(lr), loss='mse')
    callbacks_a = [
        EarlyStopping(monitor='val_loss', patience=50, verbose=1),
        ModelCheckpoint(os.path.join(ckpt_dir, 'model_complex_best.h5'), save_best_only=True, monitor='val_loss', mode='min')
    ]
    history_a = model_a.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=num_epoch, batch_size=batch_size, callbacks=callbacks_a, shuffle=False)
    print("model A min val_loss:", min(history_a.history['val_loss']))
    plot_loss_history(history_a, "Complex_Model")
elif model_type == 'simple':
    print("train simple model")
    model_b = create_simple_model()
    model_b.compile(optimizer=Adam(lr), loss='mse')
    callbacks_b = [
        EarlyStopping(monitor='val_loss', patience=50, verbose=1),
        ModelCheckpoint(os.path.join(ckpt_dir, 'model_simple_best.h5'), save_best_only=True, monitor='val_loss', mode='min')
    ]
    history_b = model_b.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=num_epoch, batch_size=batch_size, callbacks=callbacks_b, shuffle=False)
    print("model B min val_loss:", min(history_b.history['val_loss']))
    plot_loss_history(history_b, "Simple_Model")
else:
    print("train dnn model")
    model_c = create_dnn_model()
    model_c.compile(optimizer=Adam(lr), loss='mse')
    callbacks_c = [
        EarlyStopping(monitor='val_loss', patience=50, verbose=1),
        ModelCheckpoint(os.path.join(ckpt_dir, 'model_dnn_best.h5'), save_best_only=True, monitor='val_loss', mode='min')
    ]
    history_c = model_c.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=num_epoch, batch_size=batch_size, callbacks=callbacks_c, shuffle=False)
    print("model C min val_loss:", min(history_c.history['val_loss']))
    plot_loss_history(history_c, "DNN_Model")