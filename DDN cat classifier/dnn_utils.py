
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image



def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))


    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def sigmoid(Z):
    """
    Sigmoid function
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):
    """
    RELU function
    """
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    cache = Z

    return A, cache


def relu_backward(dA, cache):
    """
    Back prop for RELU
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # conversion to a correct object.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    """
    Back prop for sigmoid
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert (dZ.shape == Z.shape)

    return dZ


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- dimensions of each layer

    Returns:
    parameters -- parameters "W1", "b1", ..., "WL", "bL"
    """

    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.1
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    """
    Layer's forward prop

    Arguments:
    A -- activations from previous layer
    W -- weights
    b -- bias

    Returns:
    Z -- the input of the activation function
    cache -- a python dictionary containing "A", "W" and "b"
    """

    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache



def linear_activation_forward(A_prev, W, b, activation):
    """
    Forward prop for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer
    W -- weights
    b -- bias
    activation --  "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function
    cache -- "linear_cache" and "activation_cache"
    """

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    Forward prop for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID

    Arguments:
    X -- data
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward()
                the cache of linear_sigmoid_forward()
    """

    caches = []
    A = X
    L = len(parameters) // 2

    # [LINEAR -> RELU]*(L-1)
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation="relu")
        caches.append(cache)

    # LINEAR -> SIGMOID.
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation="sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def compute_cost(AL, Y):
    """
    Cost function

    Arguments:
    AL -- predictions
    Y -- true "label"

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))

    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return cost

def linear_backward(dZ, cache):
    """
    Back prop for a single layer

    Arguments:
    dZ -- Gradient of the cost
    cache -- tuple of values (A_prev, W, b)

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation
    dW -- Gradient of the cost with respect to W
    db -- Gradient of the cost with respect to b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Back prop for the LINEAR->ACTIVATION

    Arguments:
    dA -- post-activation gradient for current layer
    cache -- tuple of values (linear_cache, activation_cache)
    activation -- "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost
    dW -- Gradient of the cost with respect to W
    db -- Gradient of the cost with respect to b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    Back prop for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID

    Arguments:
    AL -- output of the forward prop
    Y -- true "label"
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu"
                the cache of linear_activation_forward() with "sigmoid"

    Returns:
    grads -- gradients
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # Initializing the back prop
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients.
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation="sigmoid")

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters
    grads

    Returns:
    parameters --  updated parameters
    """

    L = len(parameters) // 2

    # Update each parameter
    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate=0.075, num_iterations=3000, print_cost=False):
    """
    L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data
    Y -- true "label"
    layers_dims -- layers size
    learning_rate
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, print the cost every 100 steps

    Returns:
    parameters -- learnt parameters
    """

    costs = []

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def predict(X, y, parameters):
    """
    Predict and check accuracy

    Arguments:
    X -- examples
    parameters -- trained parameters

    Returns:
    p -- predictions
    """

    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1, m))

    # Forward prop
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == y) / m)))


    return p

def print_mislabeled_images(classes, X, y, p):
    """
    Plots mislabeled pics
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')


    plt.title("Mislabeled pictures:\n Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n True label: " + classes[y[0, index]].decode("utf-8"))
    plt.show()
