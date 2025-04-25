


# Prediction Code 


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import os
import string
import cv2

# Load trained model
model = tf.keras.models.load_model("D:\\Coding\\python\\OCR\\Model\\ocr_character_model.keras")

# Define image size
IMG_SIZE = (32, 32)

characters = list(string.digits + string.ascii_uppercase)  # ['0', '1', ..., '9', 'A', ..., 'Z']
char_to_int = {char: i for i, char in enumerate(characters)}

image_folder = "temp"




def segment_and_save_characters(image_path, save_dir=image_folder):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load image as it is (no conversion to grayscale)
    img = cv2.imread(image_path)

    # Convert to binary image based on color intensity
    # We use a threshold on the color channels to isolate characters
    # For example, we can convert each channel to binary and combine them
    _, thresh_b = cv2.threshold(img[:, :, 0], 127, 255, cv2.THRESH_BINARY_INV)
    _, thresh_g = cv2.threshold(img[:, :, 1], 127, 255, cv2.THRESH_BINARY_INV)
    _, thresh_r = cv2.threshold(img[:, :, 2], 127, 255, cv2.THRESH_BINARY_INV)

    # Combine binary masks of all channels
    thresh = cv2.bitwise_or(thresh_b, cv2.bitwise_or(thresh_g, thresh_r))

    # Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]

    # Sort bounding boxes top-to-bottom, then left-to-right
    line_threshold = 20  # group boxes into lines by y-position
    boxes = sorted(boxes, key=lambda b: (b[1] // line_threshold, b[0]))

    # Save cropped character images
    for i, (x, y, w, h) in enumerate(boxes):
        char_img = img[y:y+h, x:x+w]  # crop only from the original image
        filename = os.path.join(save_dir, f"char_{i+1:02}.png")
        cv2.imwrite(filename, char_img)

    print(f"Saved {len(boxes)} characters to '{save_dir}'")


temp_folder = "D:\\Coding\\python\\OCR\\temp"  # Replace with your actual path

# Delete all image files (e.g., .png, .jpg, .jpeg)
for file in os.listdir(temp_folder):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        os.remove(os.path.join(temp_folder, file))


# Usage
segment_and_save_characters("D:\\Coding\\python\\OCR\\Temp4.jpeg")


# Get all image file paths sorted
image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png")])


result = ''
# Loop through each image path
for i, img_path in enumerate(image_files):


    # Load and preprocess image
    img = load_img(img_path, target_size=IMG_SIZE, color_mode="grayscale")
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, 32, 32, 1)  # Reshape for model input

    # Predict character
    y_pred = model.predict(img_array)
    predicted_label = np.argmax(y_pred)  # Get the predicted class index

    # Convert label index to character
    int_to_char = {i: char for char, i in char_to_int.items()}  # Reverse dictionary
    predicted_char = int_to_char[predicted_label]

    # Display the image with the predicted character
    # plt.imshow(img, cmap="gray")
    # plt.title(f"Predicted Character: {predicted_char}")
    # plt.axis("off")
    # plt.show()

    print(f"✅ Model Prediction: {predicted_char}")
    result += predicted_char

print(f"Final Output ✅ : {result}")
