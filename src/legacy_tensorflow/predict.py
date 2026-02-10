import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# 🔧 Setări
MODEL_PATH = "models/model.h5"
IMG_SIZE = 224

# 🔥 Ordinea claselor EXACT cum a fost antrenat modelul
CLASSES = ["Bubble", "Crack", "Inclusion", "OK", "Scratch"]

# 🔇 Eliminăm warnings TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# 🔍 Încarcă modelul
print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!\n")

def predict_image(img_path):
    """Face predicția pentru o singură imagine."""
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalizare
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    class_id = np.argmax(preds)
    confidence = preds[0][class_id] * 100

    print(f"🖼️ Image: {img_path}")
    print(f"🔎 Predicted class: {CLASSES[class_id]}")
    print(f"📊 Confidence: {confidence:.2f}%\n")

# 🔄 Loop de predicții
while True:
    img_path = input("👉 Enter image path (or 'q' to quit): ")

    if img_path.lower() == "q":
        break

    if not os.path.exists(img_path):
        print("❌ File not found!\n")
        continue

    predict_image(img_path)
