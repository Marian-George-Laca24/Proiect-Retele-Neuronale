import json
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, f1_score

# =====================
# SETĂRI
# =====================
MODEL_PATH = "models/trained_model.h5"
TEST_DIR = "data/split_balanced/test"
IMG_SIZE = 224
BATCH_SIZE = 16

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================
# ÎNCĂRCARE MODEL
# =====================
print("📦 Se încarcă modelul antrenat...")
model = load_model(MODEL_PATH)
print("✅ Model încărcat cu succes!")

# =====================
# GENERATOR DATE TEST
# =====================
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# =====================
# PREDICȚII
# =====================
print("🔍 Se rulează inferența pe setul de test...")
predictions = model.predict(test_generator)

y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# =====================
# METRICI
# =====================
accuracy = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro")

print("\n📊 REZULTATE PE TEST SET:")
print(f"✔ Accuracy: {accuracy:.4f}")
print(f"✔ F1-score (macro): {f1_macro:.4f}")

# =====================
# SALVARE METRICI
# =====================
metrics = {
    "test_accuracy": float(accuracy),
    "test_f1_macro": float(f1_macro)
}

metrics_path = os.path.join(RESULTS_DIR, "test_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\n💾 Metrici salvate în: {metrics_path}")
