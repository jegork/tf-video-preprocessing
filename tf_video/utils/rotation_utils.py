import tensorflow as tf


def get_rotation_matrix(angles, image_height, image_width):
    x_offset = ((image_width - 1) - (tf.cos(angles) *
                                        (image_width - 1) - tf.sin(angles) *
                                        (image_height - 1))) / 2.0
    y_offset = ((image_height - 1) - (tf.sin(angles) *
                                        (image_width - 1) + tf.cos(angles) *
                                        (image_height - 1))) / 2.0
    num_angles = tf.shape(angles)[0]
    return tf.concat(
        values=[
            tf.cos(angles)[:, None],
            -tf.sin(angles)[:, None],
            x_offset[:, None],
            tf.sin(angles)[:, None],
            tf.cos(angles)[:, None],
            y_offset[:, None],
            tf.zeros((num_angles, 2), tf.float32),
        ],
        axis=1)