import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape should be (batch_size, timesteps, features)
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[-1],),
                                 initializer='zeros', trainable=True)
        self.V = self.add_weight(name='att_context_vector', shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs from LSTM: (batch_size, timesteps, features)

        # Apply a dense transformation to each timestep's output
        # u: (batch_size, timesteps, features)
        u = tf.tanh(tf.einsum('btd,df->btf', inputs, self.W) + self.b)

        # Calculate attention scores
        # scores: (batch_size, timesteps, 1)
        scores = tf.einsum('btf,fk->btk', u, self.V)

        # Apply softmax to get attention weights over timesteps
        # alpha: (batch_size, timesteps, 1)
        alpha = tf.nn.softmax(scores, axis=1)

        # Create the context vector as a weighted sum of the inputs
        # context_vector: (batch_size, features)
        context_vector = tf.reduce_sum(inputs * alpha, axis=1)

        return context_vector