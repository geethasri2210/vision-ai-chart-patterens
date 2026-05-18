import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Suppress TensorFlow and Python warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# Load trained model
model = load_model("chart_model.keras")   # or "chart_model.h5"
patterns = ["Head & Shoulders", "Double Bottom", "Triangle"]

def predict_chart(image_path):
    img = load_img(image_path, color_mode="grayscale", target_size=(128,128))
    arr = img_to_array(img) / 255.0
    prediction = model.predict(np.array([arr]))
    print(f"Image: {image_path} → Detected Pattern: {patterns[np.argmax(prediction)]}")

# Batch testing
test_folder = "test_charts"
if os.path.exists(test_folder):
    for file in os.listdir(test_folder):
        if file.endswith(".png") or file.endswith(".jpg"):
            predict_chart(os.path.join(test_folder, file))
else:
    print("⚠️ Folder 'test_charts' not found. Please create it and add chart images.")
