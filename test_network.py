import numpy as np

x_test = np.load('data/x_test.npy')
y_test = np.load('data/y_test.npy')


def convert_prob_into_class(a_last):
    pred = np.copy(a_last)
    pred[a_last > 0.5] = 1
    pred[a_last <= 0.5] = 0
    return pred


def cost_function(a_last, y):
    m = a_last.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(y, np.log(a_last)) + np.multiply((1 - y), np.log(1 - a_last)))
    # Make sure cost is a scalar
    cost = np.squeeze(cost)

    return cost


def get_accuracy(a_last, Y):
    pred = convert_prob_into_class(a_last)
    return (pred == Y).all(axis=0).mean()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def single_layer_forward_propagation(x, w_cur, b_cur, activation):
    # Step 1: Apply linear combination
    z = np.dot(w_cur, x) + b_cur
    # Step 2: Apply activation function
    if activation == 'relu':
        a = relu(z)
    elif activation == 'sigmoid':
        a = sigmoid(z)
    else:
        raise Exception('Not supported activation function')

    return z, a


def full_forward_propagation(x, parameters):
    # Save (z, a) at each step, which will be used for backpropagation
    caches = {'a0': x_test.T}

    a_prev = x
    length = len(parameters) // 2

    # For 1 to N-1 layers, apply relu activation function
    for i in range(1, length):
        z, a = single_layer_forward_propagation(
            a_prev,
            parameters['w' + str(i)],
            parameters['b' + str(i)],
            'relu'
        )
        caches['z' + str(i)] = z
        caches['a' + str(i)] = a
        a_prev = a

    # For last layer, apply sigmoid activation function
    z, a_last = single_layer_forward_propagation(
        a,
        parameters['w' + str(length)],
        parameters['b' + str(length)],
        'sigmoid'
    )
    caches['z' + str(length)] = z
    caches['a' + str(length)] = a_last

    return a_last, caches


# weights = initialize_parameters([x_test.shape[1], 50, 1])
weights = np.load('weights.npy', allow_pickle=True)
weights = dict(weights.flatten()[0])

# Test with testing dataset

a_last, caches = full_forward_propagation(x_test.T, weights)

# Step 3: Calculate and store cost
cost = cost_function(a_last, y_test)
# cost_history.append(cost)

accuracy = get_accuracy(a_last, y_test)
# accuracy_history.append(accuracy)

print(f'costo: {cost}')
print(f'accuracy: {accuracy}')
