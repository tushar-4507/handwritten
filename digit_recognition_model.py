import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
import cv2
import os

# Load dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Split data into features and labels
x_train = train_data.drop(columns='label')
y_train = train_data['label']

# Normalize the pixel values
x_train = x_train / 255.0
x_test = test_data / 255.0

# Reshape to 28x28x1 for CNN input
x_train = x_train.values.reshape(-1, 28, 28, 1)
x_test = x_test.values.reshape(-1, 28, 28, 1)

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, 10)

# Define the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Save the model in the newer Keras format
model.save("Model.keras")

# Predict function for a custom image
def predict_image(img_path):
    # Check if the image exists
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found at {img_path}. Please check the file path.")
    
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not open image file at {img_path}. Please check the file path and integrity.")

    # Convert the image to grayscale and resize to 28x28
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    
    # Reshape to fit the model input shape
    reshaped = resized.reshape(1, 28, 28, 1)
    
    # Normalize pixel values
    reshaped = reshaped / 255.0
    
    # Predict the digit
    prediction = model.predict(reshaped)
    predicted_digit = np.argmax(prediction)
    
    return predicted_digit

# Test the prediction function with an image
try:
    digit_prediction = predict_image('8.png')
    print(f"Predicted digit: {digit_prediction}")
except FileNotFoundError as e:
    print(e)
