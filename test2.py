import cv2
import os

def segment_and_save_characters(image_path, save_dir="temp"):
    # Create temp directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load and process image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]

    # Sort boxes: top-to-bottom, then left-to-right
    line_threshold = 20  # adjust if lines are close
    boxes = sorted(boxes, key=lambda b: (b[1] // line_threshold, b[0]))

    # Save each character image
    for i, (x, y, w, h) in enumerate(boxes):
        char_img = thresh[y:y+h, x:x+w]
        char_img = cv2.copyMakeBorder(char_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)  # padding
        filename = os.path.join(save_dir, f"char_{i+1:02}.png")
        cv2.imwrite(filename, char_img)

    print(f"Saved {len(boxes)} characters to '{save_dir}'")

# Usage
segment_and_save_characters("D:\\Coding\\python\\OCR\\Test3.webp")
