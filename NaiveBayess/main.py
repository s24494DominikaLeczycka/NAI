# Multinomial Naive Bayess
import numpy as np
from scipy.stats import multivariate_normal

class NaiveBayes:
    def __init__(self, laplace_smoothing=1):
        self.laplace_smoothing = laplace_smoothing

    def fit(self, X, y):
        self.classes, class_counts = np.unique(y, return_counts=True)
        self.priors = class_counts / len(y)

        self.mean = np.zeros((len(self.classes), X.shape[1]))
        self.var = np.zeros((len(self.classes), X.shape[1]))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[i, :] = np.mean(X_c, axis=0)
            self.var[i, :] = np.var(X_c, axis=0) + self.laplace_smoothing

    def predict(self, X):
        likelihoods = np.zeros((X.shape[0], len(self.classes)))

        for i, c in enumerate(self.classes):
            likelihoods[:, i] = multivariate_normal.pdf(X, mean=self.mean[i], cov=np.diag(self.var[i]))

        posteriors = likelihoods * self.priors
        predictions = np.argmax(posteriors, axis=1)

        return self.classes[predictions]

def main():
    # Load the data
    train_data = np.loadtxt("train.txt", delimiter=",", dtype=object)
    test_data = np.loadtxt("test.txt", delimiter=",", dtype=object)

    # Split the data into input features (X) and labels (y)
    X_train, y_train = train_data[:, :-1].astype(float), train_data[:, -1]
    X_test, y_test = test_data[:, :-1].astype(float), test_data[:, -1]

    # Train the Naive Bayes model
    nb_model = NaiveBayes()
    nb_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = nb_model.predict(X_test)

    # Calculate the accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Generate confusion matrix
    unique_labels = np.unique(np.concatenate((y_train, y_test)))
    confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

    for true_label, pred_label in zip(y_test, y_pred):
        i = np.where(unique_labels == true_label)[0][0]
        j = np.where(unique_labels == pred_label)[0][0]
        confusion_matrix[i, j] += 1

    print("\nConfusion matrix:")
    print(confusion_matrix)

if __name__ == "__main__":
    main()

