import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =========================
# CONFIGURARE GENERALĂ
# =========================
BASE_DIR = "data/split_balanced"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.0001

MODEL_OUTPUT = "models/trained_model.h5"
HISTORY_OUTPUT = "results/training_history.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# =========================
# GENERATOARE DE DATE
# =========================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

NUM_CLASSES = train_generator.num_classes
print(f"[INFO] Număr clase: {NUM_CLASSES}")
print(f"[INFO] Clase detectate: {train_generator.class_indices}")

# =========================
# DEFINIRE MODEL
# =========================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # transfer learning clasic (corect pentru Nivel 1)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# CALLBACKS
# =========================
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_OUTPUT,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# =========================
# ANTRENARE
# =========================
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stopping, checkpoint]
)

# =========================
# SALVARE ISTORIC ANTRENARE
# =========================
history_df = pd.DataFrame(history.history)
history_df.to_csv(HISTORY_OUTPUT, index=False)

print("\n✅ ANTRENARE FINALIZATĂ")
print(f"📦 Model salvat în: {MODEL_OUTPUT}")
print(f"📊 Istoric salvat în: {HISTORY_OUTPUT}")
