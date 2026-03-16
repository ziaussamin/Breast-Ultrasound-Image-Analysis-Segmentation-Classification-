import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd

# Path to model and input folder
MODEL_PATH = 'C:\\Users\\ziaus\\Downloads\\BRAC\\Semester 10\\CSE428\\Lab\\Project\\multi_task_unet_model.h5'
INPUT_FOLDER = 'C:\\Users\\ziaus\\Downloads\\BRAC\\Semester 10\\CSE428\\Lab\\Project\\Test pictures'
OUTPUT_CSV = './classification_results.csv'

# Class labels (adjust to your task)
class_names = ['Normal', 'Benign', 'Malignant']  # For multi-class classification

# Load the model
model = load_model(MODEL_PATH)

def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load in color
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    # Convert BGR to RGB
    image_resized = cv2.resize(image, target_size)
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)  # (1, 256, 256, 3)
    return image_input, image_resized

def predict_image(image_path):
    image_input, resized_image = preprocess_image(image_path)
    segmentation_pred, classification_pred = model.predict(image_input, verbose=0)

    # Classification
    if classification_pred.shape[-1] > 1:
        class_index = np.argmax(classification_pred[0])
        class_label = class_names[class_index]
        class_confidence = classification_pred[0][class_index]
    else:
        class_label = "Positive" if classification_pred[0][0] > 0.5 else "Negative"
        class_confidence = classification_pred[0][0]

    # Segmentation mask
    segmentation_mask = segmentation_pred[0, :, :, 0]
    segmentation_mask = (segmentation_mask > 0.5).astype(np.uint8)

    return resized_image, segmentation_mask, class_label, float(class_confidence)

def display_results(image, mask, filename, label, confidence):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Segmentation\nClass: {label} ({confidence:.2f})")
    plt.imshow(image, cmap='gray')
    plt.imshow(mask, cmap='Reds', alpha=0.4)
    plt.axis('off')

    plt.suptitle(f"File: {filename}", fontsize=12)
    plt.tight_layout()
    plt.show()

def batch_process(input_folder):
    results = []

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            print(f"Processing: {filename}")

            try:
                image, mask, label, confidence = predict_image(image_path)

                # Display results side-by-side
                display_results(image, mask, filename, label, confidence)

                # Save classification results
                results.append({
                    'filename': filename,
                    'class_label': label,
                    'confidence': confidence
                })

            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Save classification results to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved classification results to {OUTPUT_CSV}")

if __name__ == "__main__":
    batch_process(INPUT_FOLDER)


#pip install numpy tensorflow matplotlib pandas

#python batch_predict_visual.py
