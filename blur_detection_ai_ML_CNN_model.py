import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Data preparation: Load images and labels
def load_images(image_dir, label, img_size=(64, 64)):
    images = []
    labels = []
    for file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = cv2.resize(img, img_size)  # Resize image
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

# Load training and testing data
clear_images, clear_labels = load_images('path_to_clear_images', 0)
blurred_images, blurred_labels = load_images('path_to_blurred_images', 1)

# Combine data
X = np.concatenate((clear_images, blurred_images), axis=0)
y = np.concatenate((clear_labels, blurred_labels), axis=0)

# Normalize image data
X = X / 255.0  # Scale the pixel values to [0, 1]

# Reshape images to be compatible with the CNN input
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# Split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Output layer (binary classification)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save('blurred_vs_clear_model.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt..legend(loc='lower right')
plt.show()