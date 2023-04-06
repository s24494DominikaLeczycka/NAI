from random import shuffle

import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, train_file, test_file, learning_rate, num_epochs):
        self.train_file = train_file
        self.test_file = test_file
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
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

    def training_step(self, ground_truth, y, X):
        assert len(self.weights) == len(X)
        for i, (w, x) in enumerate(zip(self.weights, X)):
            self.weights[i] = w + (ground_truth - y) * self.learning_rate * x
        if self.weights[0] == float('inf'):
            print('Weights became infinite, exiting.')
            exit(0)
        if self.weights[0] != self.weights[0]:
            print('Weights became NaN, exiting.')
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

    def dot_product(self, v1, v2):
        sum = 0
        for i1, i2 in zip(v1, v2):
            sum += i1 * i2
        return sum

    def run_epoch(self, data, train=False, print_predictions=False):
        accuracy = 0
        if train:
            self.learning_rate *= 0.997

        for observation in data:
            predicted, y, quant_y, X = self.predict(observation[:-1])
            ground_truth = self.classes[observation[-1]]
            if train:
                self.training_step(ground_truth, y, X)
                self.weights = self.make_unit_vector(self.weights)
            elif print_predictions:
                print('true class: ', observation[-1], ', predicted: ', predicted)
            if ground_truth == quant_y:
                accuracy += 1
        return accuracy / len(data)

    def train(self):
        data = self.read_file(self.train_file)
        self.weights = data[0][:-1] + [1.0, ]
        first_class = data[0][-1]
        self.classes[first_class] = 0
        for X in data:
            if X[-1] != first_class:
                second_class = X[-1]
                break
        self.classes[second_class] = 1
        for i in range(self.num_epochs):
            train_accuracy = self.run_epoch(data, train=True)
            self.train_accuracies.append(train_accuracy)
            shuffle(data)
            self.accuracies.append(self.test())

    def test(self, print_predictions=False):
        data = self.read_file(self.test_file)
        return self.run_epoch(data, train=False, print_predictions=print_predictions)

    def predict(self, observation):
        X = observation + [-1.0,]
        y = self.dot_product(self.weights, X)
        if y > 0:
            quant_y = 1
        else:
            quant_y = 0
        predicted = list(self.classes.keys())[list(self.classes.values()).index(quant_y)]
        return predicted, y, quant_y, X

    def accuracy_plot(self):
        plt.figure(figsize=(10, 8))
        plt.title(f'Accuracies after each epoch for learning rate {str(self.learning_rate)}.')
        plt.plot(list(range(1, self.num_epochs + 1)), self.accuracies, label='Test set')
        plt.plot(list(range(1, self.num_epochs + 1)), self.train_accuracies, label='Training set')
        plt.xlabel('epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend()
        plt.show()
        for epoch_num, accuracy in enumerate(self.accuracies):
            print(f'Accuracy for {epoch_num} epoch: {accuracy}')


def main():
    learning_rate = 0.00003
    epochs_num = 100
    experiment_example2_perceptron(epochs_num, learning_rate)
    learning_rate = 0.0035
    epochs_num = 400
    p = experiment_iris_perceptron(epochs_num, learning_rate)
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
                print('Predicted value for your observation is: ', p.predict(observation)[0])
            case 2:
                learning_rate = float(input('Podaj nową stałą uczenia: '))
                experiment_example2_perceptron(epochs_num, learning_rate)
                p = experiment_iris_perceptron(epochs_num, learning_rate)


def experiment_iris_perceptron(epochs_num, learning_rate):
    print('\nPerceotron performance for data from iris_perceptron:\n')
    p = Perceptron('data\\iris_perceptron\\training.txt', 'data\\iris_perceptron\\test.txt', learning_rate, epochs_num)
    p.train()
    # draw a plot
    p.accuracy_plot()
    return p


def experiment_example2_perceptron(epochs_num, learning_rate):
    print('\nPerceotron performance for data from example2:\n')
    p = Perceptron('data\\example2\\train.txt', 'data\\example2\\test.txt', learning_rate, epochs_num)
    p.train()
    # draw a plot
    p.accuracy_plot()


if __name__ == '__main__':
    main()
