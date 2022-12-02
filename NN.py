#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 06:55:43 2019

@author: yangsong

Forked by: @cavesdev
"""

import os
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

sns.set_style("whitegrid")

#############################################################
# Load data
#############################################################


print('Cargando datos...')
X = np.load('data/x_train.npy')
y = np.load('data/y_train.npy')
x_cv = np.load('data/x_cv.npy')
y_cv = np.load('data/y_cv.npy')
x_test = np.load('data/x_test.npy')
y_test = np.load('data/y_test.npy')

print('Datos de entrenamiento: ')
print(f'X - {X.shape}')
print(f'X[0] - {X[0]}')
print(f'y - {y.shape}')
print(f'y[0] - {y[0]}')

input_size = X.shape[1]
hidden_size = 50
output_size = 1


#########################################
# Step 1: Initialize Parameters
#########################################
def initialize_parameters(layer_dim):
    if os.path.exists('weights.npy'):
        param = np.load('weights.npy', allow_pickle=True)
        return dict(param.flatten()[0])

    # np.random.seed(100)
    parameters = {}

    for i in range(1, len(layer_dim)):
        parameters['w' + str(i)] = np.random.rand(layer_dim[i], layer_dim[i - 1]) * 0.1
        parameters['b' + str(i)] = np.zeros((layer_dim[i], 1))

    return parameters


# Test
test_parameters = initialize_parameters([input_size, hidden_size, output_size])
# print(test_parameters)


#######################################
# Step 2: Forward propagation
#######################################
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


# Test
test_z, test_a = single_layer_forward_propagation(np.transpose(X), test_parameters['w1'], test_parameters['b1'], 'relu')
# print(test_z)
# print(test_a)


def full_forward_propagation(x, parameters):
    # Save (z, a) at each step, which will be used for backpropagation
    caches = {'a0': X.T}

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


# Test
test_AL, caches = full_forward_propagation(X.T, test_parameters)
# print(test_AL)


#########################################
# Step 3: Cost function
#########################################
def cost_function(a_last, y):
    m = a_last.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(y, np.log(a_last)) + np.multiply((1 - y), np.log(1 - a_last)))
    # Make sure cost is a scalar
    cost = np.squeeze(cost)

    return cost


# Test
test_cost = cost_function(test_AL, y)
# print(test_cost)


def convert_prob_into_class(a_last):
    pred = np.copy(a_last)
    pred[a_last > 0.5] = 1
    pred[a_last <= 0.5] = 0
    return pred


def get_accuracy(a_last, Y):
    pred = convert_prob_into_class(a_last)
    return (pred == Y).all(axis=0).mean()


# Test
test_y_hat = convert_prob_into_class(test_AL)
test_accuracy = get_accuracy(test_AL, y)
# print(test_y_hat)
# print(test_accuracy)


######################################
# Step 4: Backward Propagation
######################################
def sigmoid_backward_propagation(dA, z):
    sig = sigmoid(z)
    dz = dA * sig * (1 - sig)
    return dz


def relu_backward_propagation(dA, z):
    dz = np.array(dA, copy=True)
    dz[z <= 0] = 0
    return dz


def single_layer_backward_propagation(
        dA_cur,
        w_cur,
        b_cur,
        z_cur,
        A_prev,
        activation
):
    # Number of example
    m = A_prev.shape[1]

    # Part 1: Derivative for activation function
    # Select activation function
    if activation == 'sigmoid':
        backward_activation_func = sigmoid_backward_propagation
    elif activation == 'relu':
        backward_activation_func = relu_backward_propagation
    else:
        raise Exception('Not supported activation function')
    # calculate derivative
    dz_cur = backward_activation_func(dA_cur, z_cur)

    # Part 2: Derivative for linear combination
    dw_cur = np.dot(dz_cur, A_prev.T) / m
    db_cur = np.sum(dz_cur, axis=1, keepdims=True) / m
    dA_prev = np.dot(w_cur.T, dz_cur)

    return dA_prev, dw_cur, db_cur


# Test
dA_cur = - (np.divide(y, test_AL) - np.divide((1 - y), (1 - test_AL)))
dA_prev, dw_cur, db_cur = single_layer_backward_propagation(dA_cur, test_parameters['w2'], test_parameters['b2'],
                                                            caches['z2'], caches['a1'], 'sigmoid')
# print(dw_cur)
# print(db_cur)
# print(dA_prev)


def full_backward_propagation(a_last, y, caches, parameters):
    grads = {}
    length = len(caches) // 2
    m = a_last.shape[1]
    y = y.reshape(a_last.shape)

    # Step 1: Derivative for cost function
    dA_cur = - (np.divide(y, a_last) - np.divide((1 - y), (1 - a_last)))

    # Step 2: Sigmoid backward propagation for N layer
    w_cur = parameters['w' + str(length)]
    b_cur = parameters['b' + str(length)]
    z_cur = caches['z' + str(length)]
    A_prev = caches['a' + str(length - 1)]

    dA_prev, dw_cur, db_cur = single_layer_backward_propagation(dA_cur, w_cur, b_cur, z_cur, A_prev, 'sigmoid')

    grads['dw' + str(length)] = dw_cur
    grads['db' + str(length)] = db_cur

    # Step 3: relu backward propagation for 1:(N-1) layer
    for i in reversed(range(1, length)):
        dA_cur = dA_prev
        w_cur = parameters['w' + str(i)]
        b_cur = parameters['b' + str(i)]
        z_cur = caches['z' + str(i)]
        A_prev = caches['a' + str(i - 1)]

        dA_prev, dw_cur, db_cur = single_layer_backward_propagation(dA_cur, w_cur, b_cur, z_cur, A_prev, 'relu')

        grads['dw' + str(i)] = dw_cur
        grads['db' + str(i)] = db_cur

    return grads


# Test
test_grads = full_backward_propagation(test_AL, y, caches, test_parameters)
# print(test_grads['dw2'])
# print(test_grads['db2'])
# print(test_grads['dw1'])
# print(test_grads['db1'])


########################################
# Step 5 Update parameters
########################################
def update_parameters(parameters, grads, learning_rate):
    for i in (range(1, len(parameters) // 2 + 1)):
        parameters['w' + str(i)] -= grads['dw' + str(i)] * learning_rate
        parameters['b' + str(i)] -= grads['db' + str(i)] * learning_rate

    return parameters


# test
test_parameters_update = update_parameters(test_parameters, test_grads, 1)
# print(test_parameters_update['w1'])
# print(test_parameters_update['b1'])
# print(test_parameters_update['w2'])
# print(test_parameters_update['b2'])

###############################
# Create Random Dataset
###############################
# N_SAMPLES = 1000
# X, y = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=100)


#######################################
# Step 6: Train Neural Network Model
#######################################
def train_model(X, y, epoch, layer_dim, learning_rate):
    # Store historical cost
    cost_history = []
    accuracy_history = []
    epoches = []
    # Step 1: Initialize parameters
    parameters = initialize_parameters(layer_dim)

    for i in range(1, epoch):
        # Step 2: Forward propagation
        a_last, caches = full_forward_propagation(X, parameters)

        # Step 3: Calculate and store cost
        cost = cost_function(a_last, y)
        cost_history.append(cost)

        accuracy = get_accuracy(a_last, y)
        accuracy_history.append(accuracy)

        epoches.append(i)
        # Step 4: Backward propagation
        grads = full_backward_propagation(a_last, y, caches, parameters)

        # Step 5: Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            print('i=' + str(i) + ' cost = ' + str(cost))
            print('i=' + str(i) + ' accuracy = ' + str(accuracy))
            # print(parameters)

    return parameters, cost_history, accuracy_history, epoches

# Test
test_parameters, test_cost, test_accuracy, test_epoches = train_model(
    X.T,
    y,
    6000,
    [input_size, hidden_size, output_size],
    0.1
)

####################################
# Visualize cost and accuracy
####################################
Epoch = pd.DataFrame(test_epoches)
Cost = pd.DataFrame(test_cost)
Accuracy = pd.DataFrame(test_accuracy)
data = pd.concat([Epoch, Cost, Accuracy], axis=1)
data.columns = ['Epoch', 'Cost', 'Accuracy']
plt.scatter(data['Epoch'], data['Cost'])
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()

plt.scatter(data['Epoch'], data['Accuracy'], c='g')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Save weights
np.save('weights.npy', test_parameters)

# Test with validation dataset

a_last, caches = full_forward_propagation(x_cv.T, test_parameters)

# Step 3: Calculate and store cost
cost = cost_function(a_last, y_cv)
# cost_history.append(cost)

accuracy = get_accuracy(a_last, y_cv)
# accuracy_history.append(accuracy)

print(cost)
print(accuracy)