import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

# Task 1
A_tensor = torch.IntTensor(2, 3, 4)
B_tensor = torch.FloatTensor(5, 9)
C_tensor = torch.FloatTensor(12,)
D_tensor = B_tensor.view(3, 3, 5)
# print(D_tensor)

# Task 2
data = pd.read_csv("dataset.csv", encoding="utf-8")
# X = data.iloc[:, :2].values[:]
# Y = data["target"].values.reshape((-1, 1))[:]
# n_of_features = X.shape[1]


def learn(X, Y, learning_rate, number_of_epochs):
    neuron = torch.nn.Sequential(torch.nn.Linear(n_of_features, out_features=1), torch.nn.Sigmoid())
    neuron(torch.autograd.Variable(torch.FloatTensor([1, 1])))

    X = torch.autograd.Variable(torch.FloatTensor(X))
    Y = torch.autograd.Variable(torch.FloatTensor(Y))
    loss_list = []
    iter_list = []
    loss_func = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(neuron.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    for i in range(number_of_epochs):
        Y_hat = neuron(X)

        loss = loss_func(Y_hat, Y)
        loss_list.append(loss.data)
        iter_list.append(i)
        if i % (number_of_epochs/10) == 9999:
            print(f'Iter: {i} -> Loss: {loss.data}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    prob_pred = neuron(X)
    y_predicted = prob_pred > 0.5
    y_predicted = y_predicted.data.numpy().reshape(-1)
    print(y_predicted)
    plt.figure(figsize=(10, 8))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=y_predicted, cmap='brg')
    plt.title('Гавиал и крокодил', fontsize=15)
    plt.xlabel('Тон окраски', fontsize=14)
    plt.ylabel('Ширина мордочки', fontsize=14)
    plt.show()
    return loss_list


def parametric_plot(N, D, K):
    X = np.zeros((N * K, D))
    y = np.zeros(N * K, dtype='uint8')

    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    X = torch.autograd.Variable(torch.FloatTensor(X))
    y = torch.autograd.Variable(torch.LongTensor(y.astype(np.int64)))
    return X, y


def learn_parametric(X, Y, learning_rate, number_of_epochs):
    ndim_in = 2  # размерность входа (то есть количество признаков)
    ndim_out = 3  # размерность выходного слоя (то есть количество классов)
    num_epochs = number_of_epochs
    learning_rate = learning_rate
    loss_list = []

    neuron = torch.nn.Sequential(
        torch.nn.Linear(ndim_in, ndim_out),
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.RMSprop(neuron.parameters(), lr=learning_rate)

    for i in range(num_epochs):
        y_pred = neuron(X)
        loss = loss_fn(y_pred, Y)
        loss_list.append(loss.data)
        if i % 1000 == 0:
            print(f'Iter: {i} -> Loss: {loss.data}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_list, neuron


def plot_loss(lossess):
    plt.figure(figsize=(10, 8))
    plt.plot(lossess, color="m")
    plt.show()


def predict_parametric(neuron):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

    Z = neuron(torch.autograd.Variable(grid_tensor))
    Z = Z.data.numpy()
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))

    plt.contourf(xx, yy, Z, cmap=plt.cm.brg, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.brg)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$y$', fontsize=14)
    plt.show()


X, Y = parametric_plot(100, 2, 3)
loss_second, second_neuron = learn_parametric(X, Y, 0.01, 10000)
plot_loss(loss_second)
predict_parametric(second_neuron)

# first neuron
# L1loss SGD 0.01 1000000 0.091
# SmoothL1Loss SGD 0.01 1000000 0.0258
# SmoothL1Loss Adamax ==/== 2.47043226409005e-05
