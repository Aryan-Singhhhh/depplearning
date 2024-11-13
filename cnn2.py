import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt

# 1. Create synthetic data
def create_data():
    X = np.random.randn(1000, 10)  # 1000 samples, 10 features
    y = np.random.randn(1000, 1)    # 1000 samples, 1 target (regression task)
    return X, y

# 2. Define a simple deep neural network
def create_model():
    model = models.Sequential([
        layers.Dense(50, activation='relu', input_shape=(10,)),  # hidden layer with 10 features, 50 neurons
        layers.Dense(20, activation='relu'),                      # hidden layer with 20 neurons
        layers.Dense(1)                                           # output layer (regression, single output)
    ])
    return model

# 3. Train and capture loss values, showing the loss per epoch
def train_model_with_history(model, optimizer, X, y, batch_size, epochs, optimizer_name):
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    history = []

    # Training with custom loop to print loss at each epoch
    for epoch in range(epochs):
        hist = model.fit(X, y, batch_size=batch_size, epochs=1, verbose=0)
        loss = hist.history['loss'][0]
        history.append(loss)
        print(f"Epoch {epoch + 1}/{epochs} - {optimizer_name} Loss: {loss:.4f}")
    return history

# 4. Compare performance of SGD, Adam, and RMSprop
# Load Data
X, y = create_data()

# Create models for SGD, Adam, and RMSprop
model_sgd = create_model()
model_adam = create_model()
model_rmsprop = create_model()

# Optimizers
optimizer_sgd = optimizers.SGD(learning_rate=0.01)  # SGD optimizer
optimizer_adam = optimizers.Adam(learning_rate=0.01)  # Adam optimizer
optimizer_rmsprop = optimizers.RMSprop(learning_rate=0.01)  # RMSprop optimizer

# Set training parameters
batch_size = 32
epochs = 100

# Train models and capture loss history, while printing epoch iterations
print("\nTraining with SGD Optimizer")
sgd_loss = train_model_with_history(model_sgd, optimizer_sgd, X, y, batch_size, epochs, "SGD")

print("\nTraining with Adam Optimizer")
adam_loss = train_model_with_history(model_adam, optimizer_adam, X, y, batch_size, epochs, "Adam")

print("\nTraining with RMSprop Optimizer")
rmsprop_loss = train_model_with_history(model_rmsprop, optimizer_rmsprop, X, y, batch_size, epochs, "RMSprop")

# 5. Plot the loss curves for comparison
plt.plot(range(1, epochs + 1), sgd_loss, label="SGD", color='blue')
plt.plot(range(1, epochs + 1), adam_loss, label="Adam", color='orange')
plt.plot(range(1, epochs + 1), rmsprop_loss, label="RMSprop", color='green')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title('SGD vs Adam vs RMSprop Optimizer: Loss Comparison')
plt.legend()
plt.grid(True)
plt.show()
