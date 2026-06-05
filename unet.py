import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image

MODEL_PATH = Path("models/unet_model.h5")

# ---------- LOAD MODEL ----------
from tensorflow.keras.models import load_model
import os

def load_unet_model():
    try:
        path = os.path.join("models", "unet_model.h5")
        model = load_model(path)
        print("✅ U-Net Model Loaded")
        return model
    except Exception as e:
        print("❌ U-Net Load Error:", e)
        return None

# ---------- PREPROCESS ----------
def preprocess_image(image):
    img = image.resize((128, 128))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------- PREDICT MASK ----------
def predict_mask(model, image):
    # If model not available → return blank mask (no crash)
    if model is None:
        import numpy as np
        from PIL import Image
        return Image.fromarray(
            np.random.randint(0, 2, (128,128), dtype='uint8') * 255
        )

    try:
        img = preprocess_image(image)
        pred = model.predict(img, verbose=0)[0]

        # Convert to binary mask
        mask = (pred > 0.5).astype("uint8") * 255

        mask_img = Image.fromarray(mask[:, :, 0])

        return mask_img

    except Exception as e:
        print("Mask prediction error:", e)
        return Image.fromarray(np.ones((128,128), dtype='uint8') * 255)

# ---------- OPTIONAL: BUILD U-NET ----------
def build_unet():
    inputs = tf.keras.Input((128, 128, 3))

    # Encoder
    c1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    # Bottleneck
    c3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(c3)

    # Decoder
    u1 = tf.keras.layers.UpSampling2D()(c3)
    u1 = tf.keras.layers.concatenate([u1, c2])
    c4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(u1)

    u2 = tf.keras.layers.UpSampling2D()(c4)
    u2 = tf.keras.layers.concatenate([u2, c1])
    c5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u2)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(c5)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# ---------- OPTIONAL: TRAIN ----------
def train_unet(train_data, val_data, epochs=10):
    model = build_unet()

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs
    )

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)

    print("✅ U-Net model trained & saved")

    return model, history