from random import shuffle

import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, train_file, test_file, learning_rate, num_epochs):
        self.train_file = train_file
        self.test_file = test_file
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.threshold = 1
        self.accuracies = []
        self.train_accuracies = []
        self.classes = dict()

    def read_file(self, path):
        train = open(path)
        lines = []
        for line in train:
            if not line.strip(): continue
            *features, label = line.split(',')
            features = list(map(float, features))
            lines.append(features + [label.strip()])
        return lines

    def delta(self, d, y, X):
        for i, (w, x) in enumerate(zip(self.weights, X)):
            self.weights[i] = w + (d - y) * self.learning_rate * x
        if self.weights[0] == float('inf'):
            print('Weights became infinite, exiting.')
            exit(0)

    def make_unit_vector(self, vector):
        length = 0
        for elem in vector:
            length += elem ** 2
        length = length ** (1 / 2)
        for i, elem in enumerate(vector):
            elem /= length
            vector[i] = elem
        return vector

    def train(self):
        data = self.read_file(self.train_file)
        self.weights = data[0][:-1] + [self.threshold, ]
        first_class = data[0][-1]
        self.classes[first_class] = 0
        for X in data:
            if X[-1] != first_class:
                second_class = X[-1]
                break
        self.classes[second_class] = 1
        for i in range(self.num_epochs):
            a = 0.95
            # weights -> unit vector
            # w = np.array(self.weights)
            # self.weights = w / np.linalg.norm(w)
            self.weights = self.make_unit_vector(self.weights)
            train_accuracy = 0
            for observation in data:
                X = observation[:-1] + [1, ]
                y = self.dot_product(self.weights, X)
                if y >= self.threshold:
                    quant_y = 1
                else:
                    quant_y = 0
                d = self.classes[observation[-1]]
                self.delta(d, y, X)
                self.threshold = self.weights[-1]
                # print(list(self.classes.keys())[list(self.classes.values()).index(y)])
                if d == quant_y:
                    train_accuracy += 1
            self.train_accuracies.append(train_accuracy / len(data))
            shuffle(data)
            self.accuracies.append(self.test())

    def dot_product(self, v1, v2):
        sum = 0
        for i1, i2 in zip(v1, v2):
            sum += i1 * i2
        return sum

    def test(self, print_predictions=False):
        data = self.read_file(self.test_file)
        accuracy = 0
        for observation in data:
            X = observation[:-1] + [1, ]
            y = self.dot_product(self.weights, X)
            if y > self.threshold:
                y = 1
            else:
                y = 0
            predicted = list(self.classes.keys())[list(self.classes.values()).index(y)]
            if print_predictions:
                print('true class: ', observation[-1], ', predicted: ', predicted)
            if predicted == observation[-1]:
                accuracy += 1
        return accuracy / len(data)

    def perceptron(self, observation):
        X = observation
        y = self.dot_product(self.weights, X)
        if y > self.threshold:
            y = 1
        else:
            y = 0
        predicted = list(self.classes.keys())[list(self.classes.values()).index(y)]
        return predicted

    def accuracy_plot(self):
        plt.figure(figsize=(20, 10))
        plt.title('Accuracies after each epoch for learning rate ' + str(self.learning_rate))
        plt.plot(list(range(1, self.num_epochs + 1)), self.accuracies)
        plt.plot(list(range(1, self.num_epochs + 1)), self.train_accuracies)
        plt.xlabel('epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.show()


def main():
    learning_rate = 0.003
    epochs_num = 1000
    # example2
    p = Perceptron('data\\example2\\train.txt', 'data\\example2\\test.txt', learning_rate, epochs_num)
    p.train()
    # draw a plot
    p.accuracy_plot()
    # iris_perceptron
    p = Perceptron('data\\iris_perceptron\\training.txt', 'data\\iris_perceptron\\test.txt', learning_rate, epochs_num)
    p.train()
    # draw a plot
    p.accuracy_plot()
    # UI
    answer = True
    while answer:
        print('Wybierz: ')
        print('\t0, jeśli chcesz wyjść z programu')
        print('\t1, jeśli chcesz podać swoją obserwację')
        print('\t2. jeśli chcesz zmenić stałą uczenia')
        answer = int(input('Wybierz (wpisz tylko jeden znak): '))
        match answer:
            case 1:
                observation = input('Podaj atrybuty obserwacji oddzielone przecinkami: ')
                observation = list(map(float, observation.split(',')))
                print('Predicted value for your observation is: ', p.perceptron(observation))
            case 2:
                learning_rate = float(input('Podaj nową stałą uczenia: '))
                # example2
                p = Perceptron('data\\example2\\train.txt', 'data\\example2\\test.txt', learning_rate, epochs_num)
                p.train()
                # draw a plot
                p.accuracy_plot()
                # iris_perceptron
                p = Perceptron('data\\iris_perceptron\\training.txt', 'data\\iris_perceptron\\test.txt', learning_rate,
                               epochs_num)
                p.train()
                # draw a plot
                p.accuracy_plot()


if __name__ == '__main__':
    main()
