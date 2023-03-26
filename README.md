# exhibit-nn-tf-torch
An example neural network using TensorFlow and PyTorch

# Neural Networks as Function Approximators

In short, a neural network (NN) can serve as function approximator. Taking a simple function, e.g. `f(x) = sin(x)`, we can ask the question: how could we try to approximate `f(x)`? By simply passing in the input data, `x`, and output data resulting from the sine function, `f(x)`, a NN will approximate the sine function so that any new inputs passed to the NN would (approximately) return the same result as if it was passed into the sine function. Here is a plot showing a basic NN approximating `sin(x)`:

![Approximating `sin(x)`](/images/02_2x8_10000samples_1000epochs_sine.png)

This model was trained on 100,000 samples of random, uniform input data (in the range `[-10,10]`). It contains two hidden layers of 64 units each, using a softmax activation function. The final output layer was linear. The loss function used was the Mean Squared Error. In summary:

<div align="center">

| Hidden Layers | Hidden Layers Depth | Activation | Samples     | Loss | Epochs |
|:--------------:|:--------------:|:-----------:|:------------:|:------------:|:------------:|
| 2 | [ 8, 8] | Softmax | 10,000      | MSE     | 1000 |

</div>

# Implementation

The above results were achieved using the Python library `TensorFlow`. An implementation of the NN is provided using this library, along with an implementation in another popular deep learning Python library, `PyTorch`. See the related source code for examples of how to implement classic, fully connected neural networks using either library.
