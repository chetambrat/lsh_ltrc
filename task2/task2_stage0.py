import pandas as pd
import torch
from matplotlib import pyplot as plt

# Task 1
A_tensor = torch.IntTensor(2, 3, 4)
B_tensor = torch.FloatTensor(5, 9)
C_tensor = torch.FloatTensor(12,)
D_tensor = B_tensor.view(3, 3, 5)
print(D_tensor)

# Task 2
data = pd.read_csv("dataset.csv", encoding="utf-8")
X = data.iloc[:, :2].values[:]
Y = data["target"].values.reshape((-1, 1))[:]
n_of_features = X.shape[1]


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


def plot_loss(lossess):
    plt.figure(figsize=(10, 8))
    plt.plot(lossess, color="m")
    plt.show()


first_neuron = learn(X, Y, 0.01, 100000)
plot_loss(first_neuron)

# L1loss SGD 0.01 1000000 0.091
# SmoothL1Loss SGD 0.01 1000000 0.0258
# SmoothL1Loss Adamax ==/== 2.47043226409005e-05
