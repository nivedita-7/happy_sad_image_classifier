import tensorflow as tf
import numpy as np

from .config import MODEL_PATH, CLASS_NAMES
from .preprocessing import preprocess_image


model = tf.keras.models.load_model(MODEL_PATH)


def predict(image):

    img = preprocess_image(image)

    prediction = model.predict(img)

    probability = float(prediction[0][0])

    if probability < 0.5:
        label = CLASS_NAMES[0]
        confidence = (1 - probability) * 100

    else:
        label = CLASS_NAMES[1]
        confidence = probability * 100

    return label, confidence