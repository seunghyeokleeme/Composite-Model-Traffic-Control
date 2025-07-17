import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


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