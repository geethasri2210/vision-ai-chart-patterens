import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from model import build_model

# Dataset folders: Vision_AI_Project/data/{HeadAndShoulders, DoubleBottom, Triangle}
base_dir = "data"
classes = ["HeadAndShoulders", "DoubleBottom", "Triangle"]

X, y = [], []
for idx, cls in enumerate(classes):
    folder = os.path.join(base_dir, cls)
    for file in os.listdir(folder):
        img = load_img(os.path.join(folder, file), color_mode="grayscale", target_size=(128,128))
        arr = img_to_array(img) / 255.0
        X.append(arr)
        y.append(idx)

X = np.array(X)
y = to_categorical(y, num_classes=len(classes))

model = build_model()
model.fit(X, y, epochs=5, batch_size=8, validation_split=0.2)
model.save("chart_model.keras")
