# Multinomial Naive Bayess
import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class GaussianNaiveBayes:
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

class MultinomialNaiveBayes:
    def __init__(self, laplace_smoothing=1):
        self.laplace_smoothing = laplace_smoothing

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.likelihoods = {}
        self.priors = {}
        self.num_rows_given_label = {}  # Store the number of rows for each class

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = X_c.shape[0] / X.shape[0]
            self.num_rows_given_label[c] = X_c.shape[0]  # Store the number of rows for this class
            self.likelihoods[c] = {}
            for feature_index in range(X_c.shape[1]):
                self.likelihoods[c][feature_index] = {}
                unique_feature_values, feature_counts = np.unique(X_c[:, feature_index], return_counts=True)
                for value, count in zip(unique_feature_values, feature_counts):
                    self.likelihoods[c][feature_index][value] = (count + self.laplace_smoothing) / (
                                np.sum(feature_counts) + self.laplace_smoothing * len(unique_feature_values))

    def predict(self, X):
        predictions = []
        for sample in X:
            posteriors = {}
            for c in self.classes:
                prior_log = np.log(self.priors[c])
                log_likelihoods_list = []

                for feature_index in range(len(sample)):

                    if sample[feature_index] in self.likelihoods[c][feature_index]:
                        likelihood_value = self.likelihoods[c][feature_index][sample[feature_index]]
                    else:
                        num_unique_values_feature = len(self.likelihoods[c][feature_index])
                        likelihood_value = self.laplace_smoothing / (
                                    num_unique_values_feature * self.laplace_smoothing + self.num_rows_given_label[c])

                    log_likelihood = np.log(likelihood_value)
                    log_likelihoods_list.append(log_likelihood)

                posterior = np.sum(log_likelihoods_list)
                posteriors[c] = prior_log + posterior
            predictions.append(max(posteriors, key=posteriors.get))
        return predictions


def main():
    #########################
    # Gaussian Naive Bayess #
    #########################
    # Load the data
    train_data = np.loadtxt("data_iris/train.txt", delimiter=",", dtype=object)
    test_data = np.loadtxt("data_iris/test.txt", delimiter=",", dtype=object)

    # Split the data into input features (X) and labels (y)
    X_train, y_train = train_data[:, :-1].astype(float), train_data[:, -1]
    X_test, y_test = test_data[:, :-1].astype(float), test_data[:, -1]

    # Train the Naive Bayes model
    nb_model = GaussianNaiveBayes()
    nb_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = nb_model.predict(X_test)

    # Calculate the accuracy
    accuracy = np.mean(y_pred == y_test)
    print('Gaussian on Iris_data')
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

    ############################
    # Multinomial Naive Bayess #
    ############################
    # Load the data
    train_data = np.loadtxt("data_car/train.txt", delimiter=",", dtype=object)
    test_data = np.loadtxt("data_car/test.txt", delimiter=",", dtype=object)

    # Convert numpy array to DataFrame
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Initialize the encoders
    feature_encoders = [LabelEncoder() for _ in range(train_df.shape[1] - 1)]
    label_encoder = LabelEncoder()

    # Apply label encoding to each feature column
    for column in range(train_df.shape[1] - 1):
        train_df[column] = feature_encoders[column].fit_transform(train_df[column])
        test_df[column] = feature_encoders[column].transform(test_df[column])

    # Apply label encoding to the label column
    train_df[train_df.shape[1] - 1] = label_encoder.fit_transform(train_df[train_df.shape[1] - 1])
    test_df[test_df.shape[1] - 1] = label_encoder.transform(test_df[test_df.shape[1] - 1])

    # Convert DataFrame back to numpy array
    train_data = train_df.to_numpy()
    test_data = test_df.to_numpy()

    # Split the data into input features (X) and labels (y)
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    # Train the Naive Bayes model
    nb_model = MultinomialNaiveBayes()
    nb_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = nb_model.predict(X_test)

    # Calculate the accuracy
    accuracy = np.mean(y_pred == y_test)
    print('Multinomial on Car_data')
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

