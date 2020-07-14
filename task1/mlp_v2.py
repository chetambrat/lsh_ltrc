import numpy as np
import matplotlib.pyplot as plt

X = np.array([[i] for i in range(10)], dtype=int)
time = [i for i in range(10)]
Y = np.square(X, dtype=int)
arr_bias = np.ones((10, 1), dtype=int)
X = np.append(X, arr_bias, axis=1)
n = len(X)
Wi_1 = np.random.rand(2)
Wi_2 = np.random.rand(2)
Wo = np.random.rand(2)
activation_derivative = 1
learning_rate = 0.001


def activate(function_):
    if function_ < 0:
        return 0
    else:
        return 1


actual = []
for i in range(8000):
    derivative_w1_list, derivative_w2_list, derivative_w3_list, derivative_w4_list, derivative_w5_list,\
        derivative_w6_list = [], [], [], [], [], []
    rndm = np.random.randint(4)
    element_x = X[rndm]
    expected = Y[rndm]
    a_node = np.dot(element_x, Wi_1)
    b_node = np.dot(element_x, Wi_2)
    z = a_node * Wo[0] + b_node * Wo[1]
    error = expected - z
    actual.append(z)
    print(len(actual))

    Wi_1[0] = Wi_1[0] + learning_rate * error * element_x[0]
    Wi_1[1] = Wi_1[1] + learning_rate * error * element_x[1]
    Wi_2[0] = Wi_2[0] + learning_rate * error * element_x[0]
    Wi_2[1] = Wi_2[1] + learning_rate * error * element_x[1]
    Wo[0] = Wo[0] + learning_rate * error * a_node
    Wo[1] = Wo[1] + learning_rate * error * b_node
print(Wi_1)
print(actual)
plt.scatter(time, actual[10], label='actual')
plt.scatter(time, Y[10], label='theory')
plt.legend(loc='upper left')
plt.show()
plt.savefig()