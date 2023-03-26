#! env python

import torch

class BasicNN(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(BasicNN, self).__init__()

        self.name = "BasicNN"

        self.internal_layers = [
            torch.nn.Linear(input_size, 8),
            torch.nn.Softmax(),
            torch.nn.Linear(8, 8),
            torch.nn.Softmax()
        ]
        # Use Xavier initialization for the linear layers
        torch.nn.init.xavier_uniform_(self.internal_layers[0].weight)
        torch.nn.init.xavier_uniform_(self.internal_layers[2].weight)

        self.output_layer = torch.nn.Linear(8, output_size)

    def forward(self, x):
        for i in range(len(self.internal_layers)):
            layer = self.internal_layers[i]
            x = layer(x)
        return self.output_layer(x)

