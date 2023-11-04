import numpy as np
import pandas as pd

class AveragePerceptron(object):
    def __init__(self, no_of_features, epochs=10, learning_rate=1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_features)
        self.bias = 0
        self.average_weights = np.zeros(no_of_features)
        self.average_bias = 0
        self.total_count = 0

    def predict(self, x):
        return 1 if np.dot(self.average_weights, x) + self.average_bias > 0 else 0

    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(len(X)):
                if self.predict(X[i]) != y[i]:
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]
                # Add the current weights and bias to the average weights and bias
                self.total_count += 1
                self.average_weights = ((self.total_count - 1) * self.average_weights + self.weights) / self.total_count
                self.average_bias = ((self.total_count - 1) * self.average_bias + self.bias) / self.total_count

    def test(self, X, y):
        predictions = [self.predict(xi) for xi in X]
        errors = sum(int(prediction != yi) for prediction, yi in zip(predictions, y))
        average_error = errors / len(y)
        return average_error

# Read your data
train_df = pd.read_csv(r'C:\Users\Zhihao\OneDrive - University of Utah\Course\Machine Learning\hw-3\bank-note\bank-note\train.csv', header=None)
test_df = pd.read_csv(r'C:\Users\Zhihao\OneDrive - University of Utah\Course\Machine Learning\hw-3\bank-note\bank-note\test.csv', header=None)

# Get train and test data
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Create and train the Average Perceptron
perceptron = AveragePerceptron(no_of_features=X_train.shape[1])
perceptron.train(X_train, y_train)

# Test the Average Perceptron
average_test_error = perceptron.test(X_test, y_test)

# Output the learned average weight vector and bias
print(f"Learned average weight vector: {perceptron.average_weights}")
print(f"Learned average bias: {perceptron.average_bias}")

# Output the average prediction error on the test data
print(f"Average prediction error on the test dataset: {average_test_error}")

