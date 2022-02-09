import tensorflow as tf


def get_zoom_matrix(zooms, image_height, image_width):
    num_zooms = tf.shape(zooms)[0]
    # The zoom matrix looks like:
    #     [[zx 0 0]
    #      [0 zy 0]
    #      [0 0 1]]
    # where the last entry is implicit.
    # Zoom matrices are always float32.
    x_offset = ((image_width - 1.) / 2.0) * (1.0 - zooms[:, 0, None])
    y_offset = ((image_height - 1.) / 2.0) * (1.0 - zooms[:, 1, None])
    return tf.concat(
        values=[
            zooms[:, 0, None],
            tf.zeros((num_zooms, 1), tf.float32),
            x_offset,
            tf.zeros((num_zooms, 1), tf.float32),
            zooms[:, 1, None],
            y_offset,
            tf.zeros((num_zooms, 2), tf.float32),
        ],
        axis=1)
