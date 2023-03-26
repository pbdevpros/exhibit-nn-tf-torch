# exhibit-nn-tf-torch
An example neural network using TensorFlow and PyTorch. The main purpose of this is to highlight the similarities and differences in the syntax used between the two and provide a simple starting point for understanding how networks can be built, trained and evaluated.

# Neural Networks as Function Approximators

In short, a neural network (NN) can serve as function approximator. Taking a simple function, e.g. `f(x) = sin(x)`, we can ask the question: how could we try to approximate `f(x)`? By simply passing in the input data, `x`, and output data resulting from the sine function, `f(x)`, a NN will approximate the sine function so that any new inputs passed to the NN would (approximately) return the same result as if it was passed into the sine function. Here is a plot showing a basic NN approximating `sin(x)`:

![Approximating `sin(x)`](/images/02_2x8_10000samples_1000epochs_sine.png)

This model was trained on 10,000 samples of random, uniform input data (in the range `[-10,10]`), trained over 1,000 epochs. It contains two hidden layers of 8 units each, using a softmax activation function. The final output layer was linear. The loss function used was the Mean Squared Error. In summary:

<div align="center">

| Hidden Layers | Hidden Layers Depth | Activation | Samples     | Loss | Epochs |
|:--------------:|:--------------:|:-----------:|:------------:|:------------:|:------------:|
| 2 | [ 8, 8] | Softmax | 10,000      | MSE     | 1000 |

</div>

# Implementation

The above results were achieved using the Python library `TensorFlow`. An implementation of the NN is provided using this library, along with an implementation in another popular deep learning Python library, `PyTorch`. See the related source code for examples of how to implement classic, fully connected neural networks using either library. The code is structured so that each file has a corresponding file inside `torch-example` or `tf-example`.    
    Defining a neural network model in `TensorFlow` is relatively clear (see `tf-example/models.py`):  

```python
class BasicNN(tf.keras.Model):

    def __init__(self, input_size, output_size):
        # ...
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
```

Creating the same network with `PyTorch` has a similar structure, where `call` and `forward` are the exact same function:

```python
class BasicNN(torch.nn.Module):

    def __init__(self, input_size, output_size):
        # ...
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
```

The model training and evaluate have bigger differences in syntax, where `torch` requires defining the training steps in a more manual way. See the `main.py` scripts in either folders to see the training implementation. The dataset (the sine function) is created inside the `dataset.py` script. A plotting utility is in `utilities.py`
