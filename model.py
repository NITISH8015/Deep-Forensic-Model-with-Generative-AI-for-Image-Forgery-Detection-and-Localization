import tensorflow as tf
import numpy as np
from pathlib import Path

MODEL_PATH = Path("models/cnn_model.h5")

# ---------- LOAD MODEL ----------
from tensorflow.keras.models import load_model
import os

def load_cnn_model():
    try:
        path = os.path.join("models", "cnn_model.h5")
        model = load_model(path)
        print("✅ CNN Model Loaded")
        return model
    except Exception as e:
        print("❌ CNN Load Error:", e)
        return None

# ---------- PREPROCESS ----------
def preprocess_image(image):
    img = image.resize((128, 128))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------- PREDICTION ----------
def predict_image(model, image):
    if model is None:
        return "AI Model Disabled (Demo Mode)"

    try:
        img = preprocess_image(image)
        pred = model.predict(img, verbose=0)[0][0]

        confidence = float(pred)

        if pred > 0.5:
            return f"Fake ({round(confidence*100,2)}%)"
        else:
            return f"Real ({round((1-confidence)*100,2)}%)"

    except Exception as e:
        print("Prediction error:", e)
        return "Prediction Error"

# ---------- OPTIONAL: CREATE MODEL ----------
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# ---------- OPTIONAL: TRAIN MODEL ----------
def train_model(train_data, val_data, epochs=10):
    model = create_model()

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs
    )

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)

    print("✅ Model trained & saved at", MODEL_PATH)

    return model, history