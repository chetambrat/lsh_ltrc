import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
Y = np.array([[0], [0], [0], [1]])
W = np.random.rand(3)


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
            random_index = np.random.randint(4)
            x_element = x_vector[random_index]
            y_element = y_vector[random_index]
            scalar = np.dot(weights, x_element)
            error_term = y_element - self.sign(scalar)
            for j in range(len(weights)):
                weights[j] += self.learning_rate * error_term * x_element[j]
        print(weights)
        return weights

    def test_binary(self, x_vector, new_weights):
        for x in x_vector:
            result = np.dot(x, new_weights)
            print(f"{x[:2]}: {result} -> {self.sign(result)}")

    def learn_square(self, x_vector, weights):
        pass


x_and = Perceptron(None, 1e-1, 200)
w_predict = x_and.learn_binary(X, Y, W)
x_and.test_binary(X, w_predict)

