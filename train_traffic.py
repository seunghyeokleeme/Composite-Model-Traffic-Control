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

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_W', shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attention_b', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        self.v = self.add_weight(name='attention_v', shape=(input_shape[-1], 1), initializer='glorot_uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        scores = K.dot(e, self.v)
        scores = K.squeeze(scores, axis=-1)
        alpha = K.softmax(scores, axis=1)
        alpha = K.expand_dims(alpha, axis=-1)
        context_vector = K.sum(x * alpha, axis=1)
        return context_vector

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


set_seeds(42)
datasets_path = './data/traffic_volume'
if not os.path.exists(datasets_path):
    raise FileNotFoundError(f"Dataset path '{datasets_path}' does not exist. Please check the path.")

model_save_path = './models'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)


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
    plt.savefig(os.path.join(model_save_path, f'loss_curve_{model_name}.png'))
    print(f"Loss curve saved as 'loss_curve_{model_name}.png'")
    plt.show()


data_loader = TrafficDataLoader(data_path=datasets_path)

# train mode
train_X, train_Y, val_X, val_Y = data_loader.train_X, data_loader.train_Y, data_loader.val_X, data_loader.val_Y


# test mode
test_X = data_loader.test_X
test_Y = data_loader.test_Y

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


# --- 실험 선택 ---

# complex model
print("train complex model")
model_a = create_complex_model()
model_a.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
callbacks_a = [
    EarlyStopping(monitor='val_loss', patience=50, verbose=1),
    ModelCheckpoint(os.path.join(model_save_path, 'model_A_best.h5'), save_best_only=True, monitor='val_loss', mode='min')
]
history_a = model_a.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=300, batch_size=128, callbacks=callbacks_a, shuffle=False)
print("model A min val_loss:", min(history_a.history['val_loss']))
plot_loss_history(history_a, "Complex_Model")

# simple model
print("simple model")
model_b = create_simple_model()
model_b.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
callbacks_b = [
    EarlyStopping(monitor='val_loss', patience=50, verbose=1),
    ModelCheckpoint(os.path.join(model_save_path, 'model_A_best.h5'), save_best_only=True, monitor='val_loss', mode='min')
]
history_b = model_b.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=300, batch_size=128, callbacks=callbacks_b, shuffle=False)
print("model B min val_loss:", min(history_b.history['val_loss']))
plot_loss_history(history_b, "Simple_Model")