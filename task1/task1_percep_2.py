import numpy as np
import matplotlib.pyplot as plt

X = np.array([[i] for i in range(10)], dtype=int)
Y = np.square(X, dtype=int)
arr_bias = np.ones((10, 1), dtype=int)
X = np.append(X, arr_bias, axis=1)
W = np.random.rand(2)


class Perceptron:
    def __init__(self, activation, learning_rate, number_of_iterations):
        self.activation = activation
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations

    @staticmethod
    def sign(function):
        if function < 0:
            return 0
        else:
            return 1

    def learn_binary(self, x_vector, y_vector, weights):
        for i in range(self.number_of_iterations):
            random_index = np.random.randint(10)
            x_element = x_vector[random_index]
            y_element = y_vector[random_index]
            scalar = np.dot(weights, x_element)
            error_term = (y_element - scalar)
            for j in range(len(weights)):
                weights[j] += self.learning_rate * error_term * x_element[j]
        print(weights)
        return weights

    @staticmethod
    def test_binary(x_vector, new_weights):
        result_list = []
        for x in x_vector:
            result = np.dot(x, new_weights)
            result_list.append(result)
            print(f"{x[:2]}: {result}")
        return result_list


x_and = Perceptron(None, 0.01, 10000)
w_predict = x_and.learn_binary(X, Y, W)
true_list = x_and.test_binary(X, w_predict)
plt.plot(true_list, marker='o', color='blue')
plt.plot(Y, marker='x', color='cyan')
plt.show()
