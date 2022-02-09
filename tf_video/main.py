from keras import backend
import tensorflow as tf
from keras.utils.control_flow_util import smart_cond
from keras.layers.preprocessing.preprocessing_utils import ensure_tensor
import numpy as np
from .utils import *
from keras.engine.base_layer import BaseRandomLayer

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
HORIZONTAL_AND_VERTICAL = "horizontal_and_vertical"
H_AXIS = -3
W_AXIS = -2


def transform(
    video,
    transforms,
    fill_mode="reflect",
    fill_value=0.0,
    interpolation="bilinear",
    output_shape=None,
):
    if output_shape is None:
        output_shape = tf.shape(video)[1:3]
        if not tf.executing_eagerly():
            output_shape_value = tf.get_static_value(output_shape)
            if output_shape_value is not None:
                output_shape = output_shape_value

    output_shape = tf.convert_to_tensor(output_shape, tf.int32, name="output_shape")

    if not output_shape.get_shape().is_compatible_with([2]):
        raise ValueError(
            "output_shape must be a 1-D Tensor of 2 elements: "
            "new_height, new_width, instead got "
            "{}".format(output_shape)
        )

    fill_value = tf.convert_to_tensor(fill_value, tf.float32, name="fill_value")

    return tf.raw_ops.ImageProjectiveTransformV3(
        images=video,
        output_shape=output_shape,
        fill_value=fill_value,
        transforms=transforms,
        fill_mode=fill_mode.upper(),
        interpolation=interpolation.upper(),
    )


def check_fill_mode_and_interpolation(fill_mode, interpolation):
    if fill_mode not in {"reflect", "wrap", "constant", "nearest"}:
        raise NotImplementedError(
            "Unknown `fill_mode` {}. Only `reflect`, `wrap`, "
            "`constant` and `nearest` are supported.".format(fill_mode)
        )
    if interpolation not in {"nearest", "bilinear"}:
        raise NotImplementedError(
            "Unknown `interpolation` {}. Only `nearest` and "
            "`bilinear` are supported.".format(interpolation)
        )


class VideoRandomZoom(BaseRandomLayer):
    def __init__(
        self,
        height_factor,
        width_factor=None,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        **kwargs
    ):
        super(VideoRandomZoom, self).__init__(seed=seed, force_generator=True, **kwargs)
        self.height_factor = height_factor
        if isinstance(height_factor, (tuple, list)):
            self.height_lower = height_factor[0]
            self.height_upper = height_factor[1]
        else:
            self.height_lower = -height_factor
            self.height_upper = height_factor

        if abs(self.height_lower) > 1.0 or abs(self.height_upper) > 1.0:
            raise ValueError(
                "`height_factor` must have values between [-1, 1], "
                "got {}".format(height_factor)
            )

        self.width_factor = width_factor
        if width_factor is not None:
            if isinstance(width_factor, (tuple, list)):
                self.width_lower = width_factor[0]
                self.width_upper = width_factor[1]
            else:
                self.width_lower = (
                    -width_factor
                )  # pylint: disable=invalid-unary-operand-type
                self.width_upper = width_factor

            if self.width_lower < -1.0 or self.width_upper < -1.0:
                raise ValueError(
                    "`width_factor` must have values larger than -1, "
                    "got {}".format(width_factor)
                )

        check_fill_mode_and_interpolation(fill_mode, interpolation)

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed

    def call(self, inputs, training=True):
        if training is None:
            training = backend.learning_phase()

        inputs = ensure_tensor(inputs, self.compute_dtype)
        original_shape = inputs.shape
        unbatched = inputs.shape.rank == 4
        # The transform op only accepts rank 4 inputs, so if we have an unbatched
        # image, we need to temporarily expand dims to a batch.

        def random_zoomed_inputs():
            """Zoomed inputs with random ops."""
            inputs_shape = tf.shape(inputs)

            if unbatched:
                batch_size = 1
                frame_size = inputs_shape[0]
            else:
                batch_size = inputs_shape[0]
                frame_size = inputs_shape[1]
                
            img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
            img_wd = tf.cast(inputs_shape[W_AXIS], tf.float32)
            height_zoom = self._random_generator.random_uniform(
                shape=[batch_size, 1],
                minval=1.0 + self.height_lower,
                maxval=1.0 + self.height_upper,
            )
            height_zoom = tf.reshape(tf.tile(height_zoom, [1, frame_size]), [-1, 1])

            if self.width_factor is not None:
                width_zoom = self._random_generator.random_uniform(
                    shape=[batch_size, 1],
                    minval=1.0 + self.width_lower,
                    maxval=1.0 + self.width_upper,
                )
                width_zoom = tf.reshape(tf.tile(width_zoom, [1, frame_size]), [-1, 1])
            else:
                width_zoom = height_zoom
            zooms = tf.cast(
                tf.concat([width_zoom, height_zoom], axis=1), dtype=tf.float32
            )

            if not unbatched:
                flat_inputs = tf.reshape(inputs, [ batch_size * frame_size, inputs_shape[2], inputs_shape[3], inputs_shape[4] ])
            
                transformed = transform(
                    flat_inputs,
                    get_zoom_matrix(zooms, img_hd, img_wd),
                    fill_mode=self.fill_mode,
                    fill_value=self.fill_value,
                    interpolation=self.interpolation,
                )

                return tf.reshape(transformed, inputs_shape)
            else:
                return transform(
                    inputs,
                    get_zoom_matrix(zooms, img_hd, img_wd),
                    fill_mode=self.fill_mode,
                    fill_value=self.fill_value,
                    interpolation=self.interpolation,
                )


        output = smart_cond(training, random_zoomed_inputs, lambda: inputs)

        output.set_shape(original_shape)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "height_factor": self.height_factor,
            "width_factor": self.width_factor,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "interpolation": self.interpolation,
            "seed": self.seed,
        }
        base_config = super(VideoRandomZoom, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class VideoRandomRotation(BaseRandomLayer):
    def __init__(
        self,
        factor,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        **kwargs
    ):
        super(VideoRandomRotation, self).__init__(
            seed=seed, force_generator=True, **kwargs
        )
        self.factor = factor
        if isinstance(factor, (tuple, list)):
            self.lower = factor[0]
            self.upper = factor[1]
        else:
            self.lower = -factor
            self.upper = factor
        if self.upper < self.lower:
            raise ValueError(
                "Factor cannot have negative values, " "got {}".format(factor)
            )

        check_fill_mode_and_interpolation(fill_mode, interpolation)

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed

    def call(self, inputs, training=True):
        if training is None:
            training = backend.learning_phase()

        inputs = ensure_tensor(inputs, self.compute_dtype)
        original_shape = inputs.shape
        unbatched = inputs.shape.rank == 4
        # The transform op only accepts rank 4 inputs, so if we have an unbatched
        # image, we need to temporarily expand dims to a batch.

        def random_rotated_inputs():
            """Rotated inputs with random ops."""
            inputs_shape = tf.shape(inputs)
            if unbatched:
                batch_size = 1
            else:
                batch_size = inputs_shape[0]

            img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
            img_wd = tf.cast(inputs_shape[W_AXIS], tf.float32)
            min_angle = self.lower * 2.0 * np.pi
            max_angle = self.upper * 2.0 * np.pi
            angles = self._random_generator.random_uniform(
                shape=[batch_size], minval=min_angle, maxval=max_angle
            )

            if unbatched:
                return transform(
                    inputs,
                    get_rotation_matrix(angles, img_hd, img_wd),
                    fill_mode=self.fill_mode,
                    fill_value=self.fill_value,
                    interpolation=self.interpolation,
                )

            else:
                angles = tf.reshape(tf.tile(tf.reshape(angles, [-1, 1]), [1, inputs_shape[1]]), [-1])
                flat_inputs = tf.reshape(inputs, [batch_size * inputs_shape[1], inputs_shape[2], inputs_shape[3], inputs_shape[4]])
                transformed = transform(
                        flat_inputs,
                        get_rotation_matrix(angles, img_hd, img_wd),
                        fill_mode=self.fill_mode,
                        fill_value=self.fill_value,
                        interpolation=self.interpolation,
                    )

                return tf.reshape(transformed, inputs_shape)
                

        output = smart_cond(training, random_rotated_inputs, lambda: inputs)
        output.set_shape(original_shape)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "factor": self.factor,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "interpolation": self.interpolation,
            "seed": self.seed,
        }
        base_config = super(VideoRandomRotation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class VideoRandomCrop(BaseRandomLayer):
    def __init__(self, height, width, seed=None, **kwargs):
        super(VideoRandomCrop, self).__init__(
            **kwargs, autocast=False, seed=seed, force_generator=True
        )
        self.height = height
        self.width = width
        self.seed = seed

    def call(self, inputs, training=True):
        if training is None:
            training = backend.learning_phase()
        inputs = ensure_tensor(inputs, dtype=self.compute_dtype)
        input_shape = tf.shape(inputs)
        h_diff = input_shape[H_AXIS] - self.height
        w_diff = input_shape[W_AXIS] - self.width

        def random_crop():
            dtype = input_shape.dtype
            rands = self._random_generator.random_uniform([2], 0, dtype.max, dtype)
            h_start = rands[0] % (h_diff + 1)
            w_start = rands[1] % (w_diff + 1)
            return tf.map_fn(
                lambda x: tf.image.crop_to_bounding_box(
                    x, h_start, w_start, self.height, self.width
                ),
                inputs,
            )

        def resize():
            _resize = lambda x: tf.cast(
                tf.image.smart_resize(x, [self.height, self.width]), self.compute_dtype
            )
            return tf.map_fn(_resize, inputs)

        return tf.cond(
            tf.reduce_all((training, h_diff >= 0, w_diff >= 0)), random_crop, resize
        )

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[H_AXIS] = self.height
        input_shape[W_AXIS] = self.width
        return tf.TensorShape(input_shape)

    def get_config(self):
        config = {
            "height": self.height,
            "width": self.width,
            "seed": self.seed,
        }
        base_config = super(VideoRandomCrop, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class VideoRandomContrast(BaseRandomLayer):
    def __init__(self, factor, seed=None, **kwargs):
        super(VideoRandomContrast, self).__init__(
            seed=seed, force_generator=True, **kwargs
        )
        self.factor = factor
        if isinstance(factor, (tuple, list)):
            self.lower = factor[0]
            self.upper = factor[1]
        else:
            self.lower = self.upper = factor
        if self.lower < 0.0 or self.upper < 0.0 or self.lower > 1.0:
            raise ValueError(
                "Factor cannot have negative values or greater than 1.0,"
                " got {}".format(factor)
            )
        self.seed = seed
        self._random_generator = backend.RandomGenerator(seed, force_generator=True)

    def call(self, inputs, training=True):
        if training is None:
            training = backend.learning_phase()
        inputs = ensure_tensor(inputs, self.compute_dtype)

        def random_contrasted_inputs():
            seed = self._random_generator.make_seed_for_stateless_op()
            if seed is not None:
                return stateless_random_contrast(
                    inputs, 1.0 - self.lower, 1.0 + self.upper, seed=seed
                )
            else:
                return random_contrast(
                    inputs,
                    1.0 - self.lower,
                    1.0 + self.upper,
                    seed=self._random_generator.make_legacy_seed(),
                )

        output = smart_cond(training, random_contrasted_inputs, lambda: inputs)
        output.set_shape(inputs.shape)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "factor": self.factor,
            "seed": self.seed,
        }
        base_config = super(VideoRandomContrast, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class VideoRandomFlip(BaseRandomLayer):
    def __init__(self, mode=HORIZONTAL_AND_VERTICAL, seed=None, **kwargs):
        super(VideoRandomFlip, self).__init__(seed=seed, force_generator=True, **kwargs)
        self.mode = mode

        if mode == HORIZONTAL:
            self.horizontal = True
            self.vertical = False
        elif mode == VERTICAL:
            self.horizontal = False
            self.vertical = True
        elif mode == HORIZONTAL_AND_VERTICAL:
            self.horizontal = True
            self.vertical = True
        else:
            raise ValueError(
                "VideoRandomFlip layer {name} received an unknown mode "
                "argument {arg}".format(name=self.name, arg=mode)
            )
        self.seed = seed
        self._random_generator = backend.RandomGenerator(seed, force_generator=True)

    def call(self, inputs, training=True):
        if training is None:
            training = backend.learning_phase()
        inputs = ensure_tensor(inputs, self.compute_dtype)

        def random_flipped_inputs():
            flipped_outputs = inputs
            if self.horizontal:
                seed = self._random_generator.make_seed_for_stateless_op()
                if seed is not None:
                    flipped_outputs = stateless_random_flip_left_right(
                        flipped_outputs, seed=seed
                    )
                else:
                    flipped_outputs = random_flip_left_right(
                        flipped_outputs, self._random_generator.make_legacy_seed()
                    )
            if self.vertical:
                seed = self._random_generator.make_seed_for_stateless_op()
                if seed is not None:
                    flipped_outputs = stateless_random_flip_up_down(
                        flipped_outputs, seed=seed
                    )
                else:
                    flipped_outputs = random_flip_up_down(
                        flipped_outputs, self._random_generator.make_legacy_seed()
                    )
            return flipped_outputs

        output = smart_cond(training, random_flipped_inputs, lambda: inputs)
        output.set_shape(inputs.shape)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "mode": self.mode,
            "seed": self.seed,
        }
        base_config = super(VideoRandomFlip, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
