# Import necessary libraries
import tensorflow as tf
import numpy as np
import tensorflow.keras as ke
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.models import Sequential
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split  # Import for splitting the data

# 1️⃣ Load and Preprocess the MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to the range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Add a channel dimension to the images (MNIST is grayscale, so the channel size is 1)
x_train = np.expand_dims(x_train, axis=-1)  # Shape: (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, axis=-1)    # Shape: (10000, 28, 28, 1)

# One-hot encode the labels for categorical classification
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# 2️⃣ Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

# 3️⃣ Build the RNN Model
model = Sequential([
    SimpleRNN(256, input_shape=(28, 28), activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4️⃣ Set up Image Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,          # Random rotations within 10 degrees
    zoom_range=0.1,             # Random zooms up to 10%
    width_shift_range=0.1,      # Random horizontal shifts
    height_shift_range=0.1      # Random vertical shifts
)

# Fit the generator to the training data
datagen.fit(x_train)

# 5️⃣ Train the Model using Augmented Data
model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=4, validation_data=(x_val, y_val))

# 6️⃣ Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# 7️⃣ Load and Preprocess Custom Image (4.png)
# Load the image
image_path = '/content/zeroim.png'  # Path to your image file
image = cv2.imread(image_path)

# Check if the image was loaded correctly
if image is None:
    print(f"Error loading image: {image_path}")
else:
    # Resize the image to 28x28 pixels (same as MNIST format)
    image = cv2.resize(image, (28, 28))
    
    # Convert the image to grayscale (since MNIST images are grayscale)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize the image to the range [0, 1]
    image = image / 255.0
    
    # Add a channel dimension (for grayscale images)
    image = np.expand_dims(image, axis=-1)  # Shape: (28, 28, 1)
    
    # Reshape to (1, 28, 28, 1) to match the input shape expected by the model
    image = image.reshape(1, 28, 28, 1)

    # 8️⃣ Make a Prediction
    output = model.predict(image)
    predicted_class = np.argmax(output)  # Get the predicted class (0-9)

    # 9️⃣ Display Results
    print("\nPrediction Probabilities:", output)
    print(f"\nPredicted Digit: {predicted_class}")

    # Display the input image
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f'Predicted Digit: {predicted_class}')
    plt.show()
