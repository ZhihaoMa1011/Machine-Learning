import numpy as np
import pandas as pd

class VotedPerceptron(object):
    def __init__(self, no_of_features, max_epochs=10, learning_rate=1):
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weights = []
        self.mistakes = []
        self.c = []

    def predict(self, x):
        vote = 0
        for weight, count in zip(self.weights, self.c):
            vote += count * (1 if np.dot(weight, x) > 0 else -1)
        return 1 if vote > 0 else 0

    def fit(self, X, y):
        w = np.zeros(no_of_features)
        c = 1  # Initialize count
        correct_predictions = 0

        for epoch in range(self.max_epochs):
            for i in range(len(X)):
                if (np.dot(X[i], w) > 0) != y[i]:
                    # Append the current weight and count if an error is made
                    self.weights.append(w)
                    self.c.append(c)
                    w = w + (self.learning_rate * ((y[i] - (1 if np.dot(w, X[i]) > 0 else 0)) * X[i]))
                    c = 1  # Reset count
                else:
                    c += 1  # Increment count for correct prediction
                    correct_predictions += 1

            # Store the number of correct predictions with the current weight vector
            self.mistakes.append(correct_predictions)
            correct_predictions = 0  # reset counter after each epoch

        # Record the last weight vector if it has some counts accumulated
        if c > 1:
            self.weights.append(w)
            self.c.append(c)

    def test(self, X_test, y_test):
        test_predictions = [self.predict(x) for x in X_test]
        errors = np.sum(test_predictions != y_test)
        average_test_error = errors / len(y_test)
        return average_test_error

# Usage of Voted Perceptron with your data

# Adjust the paths to your datasets
train_path = r'C:\Users\Zhihao\OneDrive - University of Utah\Course\Machine Learning\hw-3\bank-note\bank-note\train.csv'
test_path = r'C:\Users\Zhihao\OneDrive - University of Utah\Course\Machine Learning\hw-3\bank-note\bank-note\test.csv'

# Read datasets
train_df = pd.read_csv(train_path, header=None)
test_df = pd.read_csv(test_path, header=None)

# Separate features and labels
X_train, y_train = train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values
X_test, y_test = test_df.iloc[:, :-1].values, test_df.iloc[:, -1].values

# Initialize the Voted Perceptron
no_of_features = X_train.shape[1]
vp = VotedPerceptron(no_of_features=no_of_features)

# Train the model
vp.fit(X_train, y_train)

# Evaluate and report on the test set
average_test_error = vp.test(X_test, y_test)
print(f'Average test error: {average_test_error}')

# Report the distinct weight vectors and their counts
print('Distinct weight vectors and their counts:')
for weight, count in zip(vp.weights, vp.c):
    print(f'Weight vector: {weight}, Count: {count}')
