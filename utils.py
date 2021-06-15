import tensorflow as tf
import numpy as np
from PIL import Image

MAX_DIMS = 250

def generate_noise_image(image, noise_ratio=0.6):
    noise_image = tf.random.uniform(tf.shape(image), 0, 1)
    return noise_image * noise_ratio + image * (1 - noise_ratio)


def preprocess_image(image):
    image = tf.image.decode_image(image, channels=3) #image_tensor
    image = tf.image.resize(image,
                          tf.cast(MAX_DIMS * tf.shape(image)[0:2] / max(tf.shape(image)[0:2]), tf.int32))

    image /= 255.0
    image = tf.expand_dims(image, 0) #первая размерность для банчей
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path) #image_raw
    return preprocess_image(image)

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)