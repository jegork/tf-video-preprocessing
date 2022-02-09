import tensorflow as tf

def random_contrast(video, lower, upper, seed=None):
    """Adjust the contrast of an image or images by a random factor.

    Equivalent to `adjust_contrast()` but uses a `contrast_factor` randomly
    picked in the interval `[lower, upper)`.

    For producing deterministic results given a `seed` value, use
    `tf.image.stateless_random_contrast`. Unlike using the `seed` param
    with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the
    same results given the same seed independent of how many times the function is
    called, and independent of global seed settings (e.g. tf.random.set_seed).

    Args:
      image: An image tensor with 3 or more dimensions.
      lower: float.  Lower bound for the random contrast factor.
      upper: float.  Upper bound for the random contrast factor.
      seed: A Python integer. Used to create a random seed. See
        `tf.compat.v1.set_random_seed` for behavior.

    Usage Example:

    >>> x = [[[1.0, 2.0, 3.0],
    ...       [4.0, 5.0, 6.0]],
    ...     [[7.0, 8.0, 9.0],
    ...       [10.0, 11.0, 12.0]]]
    >>> tf.image.random_contrast(x, 0.2, 0.5)
    <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=...>

    Returns:
      The contrast-adjusted image(s).

    Raises:
      ValueError: if `upper <= lower` or if `lower < 0`.
    """
    if upper <= lower:
        raise ValueError("upper must be > lower.")

    if lower < 0:
        raise ValueError("lower must be non-negative.")

    contrast_factor = tf.random.random_uniform([], lower, upper, seed=seed)
    return adjust_contrast(video, contrast_factor)


def stateless_random_contrast(video, lower, upper, seed):
    """Adjust the contrast of images by a random factor deterministically.

    Guarantees the same results given the same `seed` independent of how many
    times the function is called, and independent of global seed settings (e.g.
    `tf.random.set_seed`).

    Args:
      image: An image tensor with 3 or more dimensions.
      lower: float.  Lower bound for the random contrast factor.
      upper: float.  Upper bound for the random contrast factor.
      seed: A shape [2] Tensor, the seed to the random number generator. Must have
        dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)

    Usage Example:

    >>> x = [[[1.0, 2.0, 3.0],
    ...       [4.0, 5.0, 6.0]],
    ...      [[7.0, 8.0, 9.0],
    ...       [10.0, 11.0, 12.0]]]
    >>> seed = (1, 2)
    >>> tf.image.stateless_random_contrast(x, 0.2, 0.5, seed)
    <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
    array([[[3.4605184, 4.4605184, 5.4605184],
            [4.820173 , 5.820173 , 6.820173 ]],
           [[6.179827 , 7.179827 , 8.179828 ],
            [7.5394816, 8.539482 , 9.539482 ]]], dtype=float32)>

    Returns:
      The contrast-adjusted image(s).

    Raises:
      ValueError: if `upper <= lower` or if `lower < 0`.
    """
    if upper <= lower:
        raise ValueError("upper must be > lower.")

    if lower < 0:
        raise ValueError("lower must be non-negative.")

    contrast_factor = tf.random.stateless_uniform(
        shape=[], minval=lower, maxval=upper, seed=seed
    )
    return adjust_contrast(video, contrast_factor)


def adjust_contrast(video, contrast_factor):
    """Adjust contrast of RGB or grayscale images.
    This is a convenience method that converts RGB images to float
    representation, adjusts their contrast, and then converts them back to the
    original data type. If several adjustments are chained, it is advisable to
    minimize the number of redundant conversions.
    `images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
    interpreted as `[height, width, channels]`.  The other dimensions only
    represent a collection of images, such as `[batch, height, width, channels].`
    Contrast is adjusted independently for each channel of each image.
    For each channel, this Op computes the mean of the image pixels in the
    channel and then adjusts each component `x` of each pixel to
    `(x - mean) * contrast_factor + mean`.
    Usage Example:
    >>> x = [[[1.0, 2.0, 3.0],
    ...       [4.0, 5.0, 6.0]],
    ...     [[7.0, 8.0, 9.0],
    ...       [10.0, 11.0, 12.0]]]
    >>> tf.image.adjust_contrast(x, 2.)
    <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
    array([[[-3.5, -2.5, -1.5],
            [ 2.5,  3.5,  4.5]],
           [[ 8.5,  9.5, 10.5],
            [14.5, 15.5, 16.5]]], dtype=float32)>
    Args:
      images: Images to adjust.  At least 3-D.
      contrast_factor: A float multiplier for adjusting contrast.
    Returns:
      The contrast-adjusted image or images.
    """
    video = tf.convert_to_tensor(video, name="images")
    shape = video.shape
    # Remember original dtype to so we can convert back if needed

    if video.dtype not in (tf.float16, tf.float32):
        raise ValueError("Cannot adjust contrast for %r" % video.dtype)

    adjust_single = lambda x: tf.raw_ops.AdjustContrastv2(
            images=x, contrast_factor=contrast_factor
        )
    if shape.ndims == 4:
        adjust = adjust_single
    
    elif shape.ndims == 5:
        adjust = lambda x: tf.map_fn(adjust_single, x)

    adjusted = tf.map_fn(adjust, video)

    return adjusted