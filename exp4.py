import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 1, 1, 1, 1])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression training
def train_logistic_regression(X, y, learning_rate, num_iterations):
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)
    bias = 0

    for _ in range(num_iterations):
        linear_model = np.dot(X, weights) + bias
        predictions = sigmoid(linear_model)

        # Update weights and bias
        dw = (1 / num_samples) * np.dot(X.T, (predictions - y))
        db = (1 / num_samples) * np.sum(predictions - y)
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias

# Train the logistic regression model
learning_rate = 0.01
num_iterations = 1000
trained_weights, trained_bias = train_logistic_regression(X, y, learning_rate, num_iterations)

# Make predictions
def predict(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    predictions = sigmoid(linear_model)
    return [1 if p >= 0.5 else 0 for p in predictions]

# Create a scatter plot of the data
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Class 0', color='blue')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1', color='red')

# Define the decision boundary
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01))
Z = np.array(predict(np.c_[xx1.ravel(), xx2.ravel()], trained_weights, trained_bias))
Z = Z.reshape(xx1.shape)

# Plot the decision boundary
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap='coolwarm')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.legend()
plt.show()
