## Multilayer Perceptron(MLP) Classification

### Implementing the functions in `utils.py`

- To implement the activation functions `relu, identity, tanh and sigmoid` we simply use `numpy` to implement the
  respective mathematical formulae.
- The `cross_entropy` function which calculates the loss is given by `- sum (yi * log(pi))`, for all `yi ∈ Y` (true
  values) and `pi ∈ P` (predicted values). In a 2D space this result can be calculated
  by `-np.sum(np.sum(y * np.log(p), axis=1)) / y.shape[0]`. The true values are encoded using `one_hot_encoding`.

### Implementing the MLP `multilayer_perceptron.py`

#### The propagation (forward-feed) phase

- Given our training input `X`, we feed this to the hidden layer and calculate its weighted sum `Z1 = (X.wh) + bh`.
  Where `X.wh` is the dot product of the input and the weights vector `wh` for the hidden layer, and `bh` is the bias
  vector. The output `Z1` is fed to the activation function to give us `G1`.
- Subsequently, `G1` is fed to the output layer to calculate `Z2 = (G1.wo) + bo`. Where `X.wo` is the dot product of the
  input and the weights vector `wo` for the output layer, and `bo` is the bias vector. `Z2` then goes through the output
  activation function (which in this case is `softmax`), to give us `G2`. `G2` is the output of our MLP.

#### The backpropagation phase

- Naturally, the MLP has to _learn_ to predict accurate classes, it does so by minimizing its errors in the
  backpropagation phase. This is done immediately after the forward propagation phase.
- In very simple terms, we propagate the error from the last layer to the input layer by 'reversing' the order of
  operations.
- To start with, the error is calculated by a simple matrix subtraction `EO = G2 - y`. Then, we calculate the gradient
  of the weighted sum `Z2`, by passing it through the derivative of the `softmax`
  function, `SO = softmax_derivate(Z2)`. We then calculate the difference (delta) `DO = SO * EO` (scalar matrix
  multiplication).
- `DO` is back-propagated to calculate the error in the hidden layer `EH = DO.woT`, where `woT` is the transpose of the
  output layer weights vector `wo`. The gradient of the hidden layer with respect
  to `SH = hidden_activation_derivative(Z1)`. We then calculate the delta for hidden layer `DH = SH * EH`.
- After doing these calculations, we have the necessary ingredients to update the weights and biases, which are the
  essential steps that allow the MLP to make better predictions.
- This is given by `wh = XT.DH * LR`, `wo = G1T.DO * LR`, `bh = sum(DH) * LR` and `bo = sum(DO) * LR`. Where `XT` is the
  transpose of the input, `G1T` is the transpose of `G1` and `LR` is the learning rate. The learning rate essentially
  controls what percentage of weights and biases are updated.

#### The `fit` method

- This method is responsible for running the forward and back propagation phases for a number of iterations. After every
  20 iterations, the loss computed by `cross_entropy` should decrease, signifying the MLP improving its predictions.

#### The `predict` method

- Once our MLP has finished training via the `fit` method, the new testing data is passed propagated (forward) to give
  us the predictions (which are probabilities of the class labels). To calculate the class labels, we simply get
  the `argmax` for these probabilities. `np.array([np.argmax(x) for x in self.mlp_output])`.
