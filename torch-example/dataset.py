#! env python

import math
import torch

def generate_data_sinx():
    batch_size = 12500
    end = int(batch_size * 0.8)
    x = torch.FloatTensor(batch_size, 1).uniform_(-10, 10)
    y = torch.sin(x)
    return x[:end], y[:end], x[end:], y[end:] # train_x, train_y, test_x, test_y

generate_data_sinx()