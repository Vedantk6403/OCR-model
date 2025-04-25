import os
import cv2
import base64
import string
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from googletrans import Translator  # ✅ NEW IMPORT

app = Flask(__name__)

# Load model and define constants
model = tf.keras.models.load_model("D:\\Coding\\python\\OCR\\Model\\ocr_character_model.keras")
IMG_SIZE = (32, 32)
characters = list(string.digits + string.ascii_uppercase)
char_to_int = {char: i for i, char in enumerate(characters)}
int_to_char = {i: char for char, i in char_to_int.items()}
temp_folder = "D:\\Coding\\python\\OCR\\temp"

# Ensure temp folder exists
os.makedirs(temp_folder, exist_ok=True)

# ✅ Translator object
translator = Translator()


def segment_characters_from_image(img):
    _, thresh_b = cv2.threshold(img[:, :, 0], 127, 255, cv2.THRESH_BINARY_INV)
    _, thresh_g = cv2.threshold(img[:, :, 1], 127, 255, cv2.THRESH_BINARY_INV)
    _, thresh_r = cv2.threshold(img[:, :, 2], 127, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.bitwise_or(thresh_b, cv2.bitwise_or(thresh_g, thresh_r))

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = sorted(boxes, key=lambda b: (b[1] // 20, b[0]))

    chars = []
    for (x, y, w, h) in boxes:
        char_img = img[y:y+h, x:x+w]
        chars.append(char_img)
    return chars


def predict_characters(char_images):
    result = ''
    for char_img in char_images:
        gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMG_SIZE)
        img_array = img_to_array(resized) / 255.0
        img_array = img_array.reshape(1, 32, 32, 1)

        y_pred = model.predict(img_array)
        predicted_label = np.argmax(y_pred)
        predicted_char = int_to_char[predicted_label]
        result += predicted_char
    return result


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    try:
        img_data = base64.b64decode(data['image'])
        img = Image.open(BytesIO(img_data)).convert('RGB')
        img = np.array(img)
    except Exception as e:
        return jsonify({'error': 'Invalid image data', 'details': str(e)}), 400

    char_images = segment_characters_from_image(img)
    predicted_text = predict_characters(char_images)

    # ✅ Translate predicted text to Marathi
    try:
        translated = translator.translate(predicted_text, src='en', dest='mr')
        marathi_text = translated.text
        
    except Exception as e:
        marathi_text = f"Translation Error: {str(e)}"
        
    print(marathi_text)
    return jsonify({
        'predicted_text': predicted_text,
        'translated_text': marathi_text
    })


if __name__ == '__main__':
    app.run(debug=True)
