#! env python

import math
import tensorflow as tf
from utilities import plot_metric 


def generate_data_sinx():
    batch_size = 12500
    end = int(batch_size * 0.8)
    x = tf.random.uniform((batch_size,1), minval=-10, maxval=10)
    y = tf.math.sin(x)
    return x[:end], y[:end], x[end:], y[end:] # train_x, train_y, test_x, test_y