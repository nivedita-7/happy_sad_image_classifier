import numpy as np
from PIL import Image

IMG_SIZE = (256, 256)

def preprocess_image(image):
    """
    Preprocess uploaded image for prediction.
    """

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image = np.array(image)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    return image