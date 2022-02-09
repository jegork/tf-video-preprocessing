from setuptools import setup

setup(
    name='tf-video',
    version='0.0.7',
    author='Jegor Kitskerkin',
    author_email='jegor.kitskerkin@gmail.com',
    packages=['tf_video', 'tf_video.utils'],
    url='https://github.com/jegork/tf-video-preprocessing',
    license='MIT License',
    description='A library providing tools for working with videos in TensorFlow.',
    long_description='tf-video provides a convenient interface in the form of native TensorFlow layers to preprocess and augment videos right in your pipeline!',
    install_requires=[
        "tensorflow >= 2.8.0",
        "keras >= 2.8.0",
        "numpy"
    ],
)

