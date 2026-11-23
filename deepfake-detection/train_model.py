"""
OPTIONAL: train a basic deepfake classifier and save it to model/deepfake_model.h5

Dataset layout (you must download a dataset yourself, e.g. a "real vs fake faces"
set from Kaggle, and put images in these folders):
    dataset/
      real/  *.jpg
      fake/  *.jpg

Run:  python train_model.py
NOTE: This is a small beginner CNN. Accuracy depends entirely on your dataset
and will NOT match professional detectors.
"""
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (128, 128)

# NOTE: app.py scales pixels to 0-1 itself, so images are scaled to 0-1 below.
# class_names order matters: real = 0, fake = 1  (app.py expects output = P(fake))
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset", validation_split=0.2, subset="training", seed=42,
    image_size=IMG_SIZE, batch_size=32, class_names=["real", "fake"],
    label_mode="binary")
val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset", validation_split=0.2, subset="validation", seed=42,
    image_size=IMG_SIZE, batch_size=32, class_names=["real", "fake"],
    label_mode="binary")

# Scale pixels to 0-1 (same as app.py does at prediction time)
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),  # probability of FAKE
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10)
model.save("model/deepfake_model.h5")
print("Saved model/deepfake_model.h5")
