#! env python

import torch
import numpy as np

from dataset import generate_data_sinx
from utilities import plot_metric
import models as ml

def compile_and_train(model, loss, x_train, y_train, epochs):
    
    optimizer = torch.optim.Adam(model.parameters())

    # Training
    model.train() # set to training mode
    for epoch in range(epochs):

        optimizer.zero_grad()
        output = model(x_train)

        # output : [batch_size, voc_size], y_train : [batch_size] (LongTensor, not one-hot)
        loss_value = loss(output, y_train)
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss_value))

        loss_value.backward()
        optimizer.step()

def train_and_predict_basic_nn(epochs):
    # dataset
    (x_train, y_train, x_test, y_test) = generate_data_sinx()

    # define model 
    model = ml.BasicNN(x_train.shape[1], y_train.shape[1])
    loss = torch.nn.MSELoss()

    # training - note: history not recorded...
    compile_and_train(model, loss, x_train, y_train, epochs)
    print("Evaluating model: " + model.name)

    # evaluation
    model.eval()
    predict = model(x_test).data.max(1, keepdim=True)[1]
    l = loss(predict, y_test)
    print("Achieved loss: {}".format(l))

    # prediction
    x = torch.linspace(-5, 5, 100)
    y = model(x.reshape(-1, 1)).detach().numpy() # we need to detach to get the values without a grad
    y1 = torch.sin(x)
    plot_metric([x, x], [y, y1], "Plot of neural net approximating sin (x)", ["nn(x)", "sin(x)"])

if __name__ == '__main__':
    epochs = 1000
    train_and_predict_basic_nn(epochs)