import kagglehub
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
import string

data = []

training_data = 'D:\\Coding\\python\\OCR\\standard-ocr-dataset\\versions\\data\\training_data'
label = 'D:\\Coding\\python\\OCR\\standard-ocr-dataset\\versions\\data\\training_data\\0'

for label in os.listdir(training_data):
    label_path = os.path.join(training_data, label)

    if os.path.isdir(label_path):
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            data.append((img_path, label))  # Store image path and label

df = pd.DataFrame(data, columns=["image_path", "label"])
characters = list(string.digits + string.ascii_uppercase)  # ['0', '1', ..., '9', 'A', ..., 'Z']
char_to_int = {char: i for i, char in enumerate(characters)}

num_classes = len(char_to_int)

IMG_SIZE = (32, 32)

def load_images(df):
    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = row["image_path"]
        label = row["label"]  
        label_int = char_to_int[label]
        img = load_img(img_path, target_size=IMG_SIZE, color_mode="grayscale")
        img_array = img_to_array(img) / 255.0  # Normalize
        images.append(img_array)
        labels.append(label_int)
    print(labels)
    return np.array(images), np.array(labels)

X_train, y_train = load_images(df)
X_train = X_train.reshape(-1, 32, 32, 1) 
y_train = to_categorical(y_train, num_classes)

print("Unique training labels:", np.unique(y_train, return_counts=True))

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Output layer with softmax activation
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

from sklearn.model_selection import train_test_split
y_int_labels = np.argmax(y_train, axis=1)  
X_train, X_val, y_train_int, y_val_int = train_test_split(X_train, y_int_labels, test_size=0.2, stratify=y_int_labels, random_state=42)
y_train = to_categorical(y_train_int, num_classes)
y_val = to_categorical(y_val_int, num_classes)
print("Training labels distribution:", np.unique(y_train_int, return_counts=True))
print("Validation labels distribution:", np.unique(y_val_int, return_counts=True))

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(32, 32, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Output layer
])
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)

history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=40,
                    validation_data=(X_val, y_val))

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save("D:\\Coding\\python\\OCR\\Model\\ocr_character_model.keras")

import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_SIZE = (32, 32)

testing_data_path = "D:\\Coding\\python\\OCR\\standard-ocr-dataset\\versions\\data\\testing_data"

def load_test_images(testing_data_path):
    test_images = []
    test_filenames = []
    for label in sorted(os.listdir(testing_data_path)):  # Sort to match training order
        label_path = os.path.join(testing_data_path, label)
        if os.path.isdir(label_path):  # Ensure it's a folder
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                img = load_img(img_path, target_size=IMG_SIZE, color_mode="grayscale")
                img_array = img_to_array(img) / 255.0  # Normalize
                test_images.append(img_array)
                test_filenames.append(img_path)  # Store filename for reference
    return np.array(test_images), test_filenames

X_test, test_filenames = load_test_images(testing_data_path)

X_test = X_test.reshape(-1, 32, 32, 1)

y_pred = model.predict(X_test)

y_pred_labels = np.argmax(y_pred, axis=1)

int_to_char = {i: char for char, i in char_to_int.items()}  # Reverse dictionary
predicted_chars = [int_to_char[i] for i in y_pred_labels]

for i in range(1000):  # Show 10 sample predictions
    print(f"Image: {test_filenames[i]} → Predicted Character: {predicted_chars[i]}") 
    
import random

y_pred = model.predict(X_test)

y_pred_labels = np.argmax(y_pred, axis=1)

int_to_char = {i: char for char, i in char_to_int.items()}  # Reverse dictionary
predicted_chars = [int_to_char[i] for i in y_pred_labels]
random_indices = random.sample(range(len(X_test)), 1000)

for i in random_indices:
    print(f"Image: {test_filenames[i]} → Predicted Character: {predicted_chars[i]}")

import numpy as np

actual_labels = [os.path.basename(os.path.dirname(f)) for f in test_filenames]
actual_labels = np.array(actual_labels, dtype=str)
predicted_chars = np.array(predicted_chars, dtype=str)
actual_labels = np.char.upper(actual_labels)  # Convert to uppercase
predicted_chars = np.char.upper(predicted_chars)
correct = np.sum(predicted_chars == actual_labels)
total = len(actual_labels)
accuracy = (correct / total) * 100

print(f"✅ Test Accuracy: {accuracy:.2f}%")
import matplotlib.pyplot as plt
import random
random_indices = random.sample(range(len(X_test)), 1000)
fig, axes = plt.subplots(10, 10, figsize=(20, 20))
for idx, ax in zip(random_indices, axes.flatten()):
    ax.imshow(X_test[idx].reshape(32, 32), cmap='gray')
    ax.set_title(f"Pred: {predicted_chars[idx]}")
    ax.axis("off")

plt.tight_layout()
plt.show()