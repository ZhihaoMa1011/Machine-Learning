import numpy as np
import pandas as pd

class Perceptron(object):
    def __init__(self, no_of_inputs, epoch=10, learning_rate=1):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels):
        for _ in range(self.epoch):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
    
    def test(self, test_inputs, test_labels):
        total_error = 0
        for inputs, label in zip(test_inputs, test_labels):
            prediction = self.predict(inputs)
            total_error += abs(label - prediction)  # Absolute error
        average_error = total_error / len(test_labels)
        return average_error

# Load datasets
train_df = pd.read_csv(r'C:\Users\Zhihao\OneDrive - University of Utah\Course\Machine Learning\hw-3\bank-note\bank-note\train.csv', header=None)  # adjust path as needed
test_df = pd.read_csv(r'C:\Users\Zhihao\OneDrive - University of Utah\Course\Machine Learning\hw-3\bank-note\bank-note\test.csv', header=None)  # adjust path as needed

# Assuming the last column is the label
training_inputs = train_df.iloc[:, :-1].values
training_labels = train_df.iloc[:, -1].values
test_inputs = test_df.iloc[:, :-1].values
test_labels = test_df.iloc[:, -1].values

# Determine the number of features (assuming all features are numerical and no missing values)
no_of_inputs = training_inputs.shape[1]

# Create the Perceptron object
perceptron = Perceptron(no_of_inputs=no_of_inputs)

# Train the Perceptron
perceptron.train(training_inputs, training_labels)

# Report the learned weight vector
print("Learned weight vector:", perceptron.weights)

# Calculate the average prediction error on the test dataset
average_prediction_error = perceptron.test(test_inputs, test_labels)
print("Average prediction error on the test dataset:", average_prediction_error)
