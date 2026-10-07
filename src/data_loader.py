import tensorflow as tf

IMG_HEIGHT = 256
IMG_WIDTH = 256


def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, tf.float32)

    input_image = image[:, :256, :]
    target_image = image[:, 256:, :]

    return input_image, target_image


def resize_images(input_image, target_image):
    input_image = tf.image.resize(
        input_image,
        [286, 286],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    target_image = tf.image.resize(
        target_image,
        [286, 286],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    return input_image, target_image


def random_crop(input_image, target_image):
    stacked_image = tf.stack(
        [input_image, target_image],
        axis=0
    )

    cropped_image = tf.image.random_crop(
        stacked_image,
        size=[2, IMG_HEIGHT, IMG_WIDTH, 3]
    )

    return cropped_image[0], cropped_image[1]


def random_jitter(input_image, target_image):
    input_image, target_image = resize_images(
        input_image,
        target_image
    )

    input_image, target_image = random_crop(
        input_image,
        target_image
    )

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        target_image = tf.image.flip_left_right(target_image)

    return input_image, target_image


def normalize(input_image, target_image):
    input_image = (input_image / 127.5) - 1
    target_image = (target_image / 127.5) - 1

    return input_image, target_image


def preprocess_image(image_file):
    input_image, target_image = load_image(image_file)

    input_image, target_image = random_jitter(
        input_image,
        target_image
    )

    return normalize(input_image, target_image)