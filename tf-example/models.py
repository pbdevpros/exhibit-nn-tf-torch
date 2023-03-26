#! env python

import tensorflow as tf

class BasicNN(tf.keras.Model):

    def __init__(self, input_size, output_size):
        super(BasicNN, self).__init__()
        self.internal_layers = [
            tf.keras.layers.Dense(8, kernel_initializer=tf.keras.initializers.GlorotUniform(), activation=tf.keras.activations.softmax),
            tf.keras.layers.Dense(8, kernel_initializer=tf.keras.initializers.GlorotUniform(), activation=tf.keras.activations.softmax)
        ]
        self.output_layer = tf.keras.layers.Dense(output_size, kernel_initializer=tf.keras.initializers.GlorotUniform()) # no activation

    def call(self, x):
        for i in range(len(self.internal_layers)):
            layer = self.internal_layers[i]
            x = layer(x)
        return self.output_layer(x)
