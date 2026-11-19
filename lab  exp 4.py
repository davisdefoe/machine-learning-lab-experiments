import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target.reshape(-1, 1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize network parameters
np.random.seed(42)
input_size = X_train.shape[1]
hidden_size = 10
output_size = 1

W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# Training using backpropagation
learning_rate = 0.01
epochs = 2000

for epoch in range(epochs):
    # Forward propagation
    hidden = sigmoid(X_train @ W1 + b1)
    output = sigmoid(hidden @ W2 + b2)

    # Backpropagation
    error = output - y_train
    d_output = error

    d_hidden = (d_output @ W2.T) * hidden * (1 - hidden)

    # Update weights and biases
    W2 -= learning_rate * (hidden.T @ d_output) / len(X_train)
    b2 -= learning_rate * np.mean(d_output, axis=0, keepdims=True)

    W1 -= learning_rate * (X_train.T @ d_hidden) / len(X_train)
    b1 -= learning_rate * np.mean(d_hidden, axis=0, keepdims=True)

# Testing
hidden = sigmoid(X_test @ W1 + b1)
output = sigmoid(hidden @ W2 + b2)

predictions = (output >= 0.5).astype(int)
accuracy = accuracy_score(y_test, predictions)

print("Actual values:", y_test[:10].ravel())
print("Predicted values:", predictions[:10].ravel())
print("Test Accuracy: {:.2f}%".format(accuracy * 100))
