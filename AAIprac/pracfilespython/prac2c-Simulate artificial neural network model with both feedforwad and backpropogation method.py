import numpy as np

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the Sigmoid Function for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# ANN class to simulate feedforward and backpropagation
class ArtificialNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        # Initialize weights randomly
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)

        # Initialize biases randomly
        self.bias_hidden = np.random.rand(1, hidden_size)
        self.bias_output = np.random.rand(1, output_size)

        # Set the learning rate
        self.learning_rate = learning_rate

    # Feedforward process
    def feedforward(self, X):
        # Hidden layer activation
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        # Output layer activation
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)

        return self.output

    # Backpropagation process
    def backpropagation(self, X, y):
        # Error at the output layer
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        # Error at the hidden layer
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update the weights and biases using the deltas
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    # Train the neural network
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            # Feedforward
            self.feedforward(X)
            # Backpropagation
            self.backpropagation(X, y)
            # Print loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                loss = np.mean(np.square(y - self.output))
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}')

# Example usage
if __name__ == "__main__":
    # Input dataset (XNOR problem)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    # Output dataset (XNOR output)
    y = np.array([[1],
                  [0],
                  [0],
                  [1]])

    # Parameters
    input_size = X.shape[1]  # 2 features in input
    hidden_size = 2          # 2 neurons in hidden layer
    output_size = 1          # 1 output neuron (binary classification)

    # Create the neural network
    ann = ArtificialNeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.5)

    # Train the neural network
    ann.train(X, y, epochs=10000)

    # Test the neural network
    output = ann.feedforward(X)
    print("\nPredicted Output after training:")
    print(output)

print("Deep Marathe -53004230016")