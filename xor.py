#updating weights and bias
import numpy as np

# Define unit step function (activation function)
def unitStep(v):
    return 1 if v >= 0 else 0

# Define the perceptron model (with weight and bias updates)
class Perceptron:
    def _init_(self, n_iterations=10, learning_rate=0.1):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    # Perceptron prediction (linear output + activation)
    def perceptronModel(self, x):
        v = np.dot(x, self.weights) + self.bias
        y = unitStep(v)
        return y

    # Train the perceptron (adjusting weights and bias)
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training process
        for _ in range(self.n_iterations):
            for i in range(n_samples):
                # Calculate linear output and prediction
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_predicted = unitStep(linear_output)

                # Calculate the update
                update = self.learning_rate * (y[i] - y_predicted)

                # Update the weights and bias
                self.weights += update * X[i]
                self.bias += update

    # Logic gate functions using learned weights
    def AND_logicFunction(self, x):
        return self.perceptronModel(x)

    def OR_logicFunction(self, x):
        return self.perceptronModel(x)

    def NOT_logicFunction(self, x):
        # Special case since NOT gate is unary (single input)
        w = np.array([-1])
        b = 0.5
        v = np.dot(w, x) + b
        return unitStep(v)

    def XOR_logicFunction(self, x):
        y1 = self.AND_logicFunction(x)
        y2 = self.OR_logicFunction(x)
        y3 = self.NOT_logicFunction(np.array([y1]))
        final_x = np.array([y2, y3])
        finalOutput = self.AND_logicFunction(final_x)
        return finalOutput

# Training data for AND, OR, and XOR gates
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Labels for AND gate
y_AND = np.array([0, 0, 0, 1])  # AND truth table
# Labels for OR gate
y_OR = np.array([0, 1, 1, 1])   # OR truth table

# Initialize Perceptron model
perceptron = Perceptron(n_iterations=10, learning_rate=0.1)

# Train for AND logic
print("Training for AND gate...")
perceptron.fit(X_train, y_AND)

# Test XOR logic using trained AND and OR gates
test1 = np.array([0, 1])
test2 = np.array([1, 1])
test3 = np.array([0, 0])
test4 = np.array([1, 0])

print("XOR({},{})={}".format(0,1,perceptron.XOR_logicFunction(test1)))
print("XOR({},{})={}".format(1,1,perceptron.XOR_logicFunction(test2)))
print("XOR({},{})={}".format(0,0,perceptron.XOR_logicFunction(test3)))
print("XOR({},{})={}".format(1,0,perceptron.XOR_logicFunction(test4)))
