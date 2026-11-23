"""
Train a deepfake classifier using MobileNetV2 (transfer learning).

Expects this layout (folder names in CAPITALS, as in your dataset):
    dataset/train/REAL, dataset/train/FAKE
    dataset/test/REAL,  dataset/test/FAKE

Saves the model to: model/deepfake_model.h5
"""
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------- Settings (change these if needed) ----------
DATASET_DIR = "dataset"     # folder that contains "train" and "test"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5
MAX_TRAIN_BATCHES = 300     # 300 x 32 = ~9,600 images per epoch. Keeps training fast.
                            # Set to None to use ALL training images.
MAX_TEST_BATCHES = 60       # ~1,900 test images. Set to None to use all.

# ---------- Load images from the folders ----------
# class_names=["REAL", "FAKE"] makes REAL = 0 and FAKE = 1,
# so the model outputs the probability of FAKE (this is what app.py expects).
common = dict(image_size=IMG_SIZE, batch_size=BATCH_SIZE,
              class_names=["REAL", "FAKE"], label_mode="binary")

train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_DIR}/train", shuffle=True, seed=42, **common)
test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_DIR}/test", shuffle=False, **common)

# Use only part of the data so training finishes quickly
if MAX_TRAIN_BATCHES:
    train_ds = train_ds.take(MAX_TRAIN_BATCHES)
if MAX_TEST_BATCHES:
    test_ds = test_ds.take(MAX_TEST_BATCHES)

# Scale pixels to 0-1 (app.py does the same when predicting)
train_ds = train_ds.map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE)

# ---------- Build the model ----------
# Pre-trained MobileNetV2 (downloads ImageNet weights on first run, needs internet)
base = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3), include_top=False, weights="imagenet")
base.trainable = False  # keep the pre-trained knowledge, train only our new layers

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(2.0, offset=-1.0),   # converts 0-1 input into the -1..1 range MobileNetV2 expects
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),  # probability of FAKE
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ---------- Train (the test folder is used to check progress each epoch) ----------
model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)

# ---------- Final accuracy on the test images ----------
loss, accuracy = model.evaluate(test_ds)
print(f"\nTest accuracy: {accuracy * 100:.1f}%")

# ---------- Save ----------
model.save("model/deepfake_model.h5")
print("Saved model/deepfake_model.h5")