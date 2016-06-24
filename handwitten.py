# -*- coding: cp949 -*-
import os
import gzip
import numpy as np
from urllib import urlretrieve
np.seterr(all = 'ignore')


def feedForward(weight1, weight2, X, yvector):
    z2 = np.dot(X, np.transpose(weight1))
    a2 = sigmoid(z2)
    z3 = np.dot(a2, np.transpose(weight2))
    a3 = sigmoid(z3)
    h = a3

    temp = h - yvector

    J = np.sum(np.multiply(temp, temp)) / 2

    print "error : ", J

    return h, a2


def backPropagation(h, X, yvector, weight1, weight2, a2):
    alpha = 0.6
    n = len(h)

    delta3 = h - yvector

    ones = np.ones((len(a2), len(a2[0])))
    sigmoidGradient = np.multiply(sigmoid(a2), ones - sigmoid(a2))
    delta2 = np.multiply(np.dot(delta3, weight2), sigmoidGradient)

    theta1Grad = np.dot(delta2.T, X) / n
    theta2Grad = np.dot(delta3.T, a2) / n

    weight1 -= alpha * theta1Grad
    weight2 -= alpha * theta2Grad
    return weight1, weight2


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def load_mnist_set():
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        urlretrieve(source + filename, filename)

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    return X_train, y_train, X_test, y_test


inputLayerSize = 784  #28*28
hiddenLayerSize = 100
outputLayerSize = 10

weight1 = np.random.uniform(-1, 1, (hiddenLayerSize, inputLayerSize))
weight2 = np.random.uniform(-1, 1, (outputLayerSize, hiddenLayerSize))

print("Loading data...")
X_train, y_train, X_test, y_test = load_mnist_set()
print("Loading complete!")

X_train = np.reshape(X_train, (len(X_train), inputLayerSize))
y_train = np.reshape(y_train, (len(y_train), 1))
X_test = np.reshape(X_test, (len(X_test), inputLayerSize))
y_test = np.reshape(y_test, (len(y_test), 1))

n = len(X_train)
yvector = np.zeros((n, outputLayerSize))    # make y vector
for i in range(n):
    yvector[i][y_train[i]] = 1

for i in range(20):
    h, a2 = feedForward(weight1, weight2, X_train, yvector)
    weight1, weight2 = backPropagation(h, X_train, yvector, weight1, weight2, a2)

n = len(X_test)
yvector = np.zeros((n, outputLayerSize))    # make y vector
for i in range(n):
    yvector[i][y_test[i]] = 1

h, a2 = feedForward(weight1, weight2, X_test, yvector)

correct = 0.0
for i in range(n):
    max = 0
    index = 0
    for j in range(outputLayerSize):
        if max < h[i][j]:
            max = h[i][j]
            index = j
    print y_test[i][0], "->", index
    if y_test[i][0] == index:
        correct += 1
print "정답률 : ", correct / n