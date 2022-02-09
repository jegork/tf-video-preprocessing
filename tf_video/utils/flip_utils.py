import tensorflow as tf
import functools


def fix_image_flip_shape(image, result):
    """Set the shape to 3 dimensional if we don't know anything else.
    Args:
      image: original image size
      result: flipped or transformed image
    Returns:
      An image whose shape is at least (None, None, None).
    """

    video_shape = image.get_shape()
    if video_shape == tf.TensorShape(None):
        result.set_shape([None, None, None, None])
    else:
        result.set_shape(video_shape)
    return result


def _random_flip(video, flip_index, random_func):
    video = tf.convert_to_tensor(video)
    shape = video.shape

    def f_rank4():
        uniform_random = random_func(shape=[], minval=0, maxval=1.0)
        mirror_cond = tf.less(uniform_random, 0.5)
        result = tf.cond(
            mirror_cond,
            lambda: tf.reverse(video, [flip_index]),
            lambda: video,
        )
        return fix_image_flip_shape(video, result)

    def f_rank5():
        batch_size = tf.shape(video)[0]
        uniform_random = random_func(shape=[batch_size], minval=0, maxval=1.0)
        flips = tf.round(tf.reshape(uniform_random, [batch_size, 1, 1, 1, 1]))
        flips = tf.cast(flips, video.dtype)
        flipped_input = tf.reverse(video, [flip_index + 1])
        return flips * flipped_input + (1 - flips) * video

    
    if shape.ndims is None:
        rank = tf.rank(video)
        return tf.cond(tf.equal(rank, 4), f_rank4, f_rank5)
    if shape.ndims == 4:
        return f_rank4()
    elif shape.ndims == 5:
        return f_rank5()
    else:
        raise ValueError(
            "'video' (shape %s) must have either 3 or 4 dimensions." % shape
        )


def random_flip_up_down(video, seed=None):
    """Randomly flips an image vertically (upside down).
    With a 1 in 2 chance, outputs the contents of `image` flipped along the first
    dimension, which is `height`.  Otherwise, output the image as-is.
    When passing a batch of images, each image will be randomly flipped
    independent of other images.
    Example usage:
    >>> image = np.array([[[1], [2]], [[3], [4]]])
    >>> tf.image.random_flip_up_down(image, 3).numpy().tolist()
    [[[3], [4]], [[1], [2]]]
    Randomly flip multiple images.
    >>> images = np.array(
    ... [
    ...     [[[1], [2]], [[3], [4]]],
    ...     [[[5], [6]], [[7], [8]]]
    ... ])
    >>> tf.image.random_flip_up_down(images, 4).numpy().tolist()
    [[[[3], [4]], [[1], [2]]], [[[5], [6]], [[7], [8]]]]
    For producing deterministic results given a `seed` value, use
    `tf.image.stateless_random_flip_up_down`. Unlike using the `seed` param
    with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the
    same results given the same seed independent of how many times the function is
    called, and independent of global seed settings (e.g. tf.random.set_seed).
    Args:
      image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
        of shape `[height, width, channels]`.
      seed: A Python integer. Used to create a random seed. See
        `tf.compat.v1.set_random_seed` for behavior.
    Returns:
      A tensor of the same type and shape as `image`.
    Raises:
      ValueError: if the shape of `image` not supported.
    """
    random_func = functools.partial(tf.random.uniform, seed=seed)
    return _random_flip(video, 1, random_func)


def random_flip_left_right(video, seed=None):
    """Randomly flip an image horizontally (left to right).
    With a 1 in 2 chance, outputs the contents of `image` flipped along the
    second dimension, which is `width`.  Otherwise output the image as-is.
    When passing a batch of images, each image will be randomly flipped
    independent of other images.
    Example usage:
    >>> image = np.array([[[1], [2]], [[3], [4]]])
    >>> tf.image.random_flip_left_right(image, 5).numpy().tolist()
    [[[2], [1]], [[4], [3]]]
    Randomly flip multiple images.
    >>> images = np.array(
    ... [
    ...     [[[1], [2]], [[3], [4]]],
    ...     [[[5], [6]], [[7], [8]]]
    ... ])
    >>> tf.image.random_flip_left_right(images, 6).numpy().tolist()
    [[[[2], [1]], [[4], [3]]], [[[5], [6]], [[7], [8]]]]
    For producing deterministic results given a `seed` value, use
    `tf.image.stateless_random_flip_left_right`. Unlike using the `seed` param
    with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the
    same results given the same seed independent of how many times the function is
    called, and independent of global seed settings (e.g. tf.random.set_seed).
    Args:
      image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
        of shape `[height, width, channels]`.
      seed: A Python integer. Used to create a random seed. See
        `tf.compat.v1.set_random_seed` for behavior.
    Returns:
      A tensor of the same type and shape as `image`.
    Raises:
      ValueError: if the shape of `image` not supported.
    """
    random_func = functools.partial(tf.random.uniform, seed=seed)
    return _random_flip(video, 2, random_func)


def stateless_random_flip_left_right(video, seed):
    """Randomly flip an image horizontally (left to right) deterministically.
    Guarantees the same results given the same `seed` independent of how many
    times the function is called, and independent of global seed settings (e.g.
    `tf.random.set_seed`).
    Example usage:
    >>> image = np.array([[[1], [2]], [[3], [4]]])
    >>> seed = (2, 3)
    >>> tf.image.stateless_random_flip_left_right(image, seed).numpy().tolist()
    [[[2], [1]], [[4], [3]]]
    Args:
      image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
        of shape `[height, width, channels]`.
      seed: A shape [2] Tensor, the seed to the random number generator. Must have
        dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
    Returns:
      A tensor of the same type and shape as `image`.
    """
    random_func = functools.partial(tf.random.stateless_uniform, seed=seed)
    return _random_flip(video, 2, random_func)


def stateless_random_flip_up_down(video, seed):
    """Randomly flip an image vertically (upside down) deterministically.
    Guarantees the same results given the same `seed` independent of how many
    times the function is called, and independent of global seed settings (e.g.
    `tf.random.set_seed`).
    Example usage:
    >>> image = np.array([[[1], [2]], [[3], [4]]])
    >>> seed = (2, 3)
    >>> tf.image.stateless_random_flip_up_down(image, seed).numpy().tolist()
    [[[3], [4]], [[1], [2]]]
    Args:
      image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
        of shape `[height, width, channels]`.
      seed: A shape [2] Tensor, the seed to the random number generator. Must have
        dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
    Returns:
      A tensor of the same type and shape as `image`.
    """
    random_func = functools.partial(tf.random.stateless_uniform, seed=seed)
    return _random_flip(video, 1, random_func)
