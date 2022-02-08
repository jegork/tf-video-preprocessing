# tf-video-preprocessing
TensorFlow video preprocessing layers

While working on a video classification model, I was in need of a video augmentation layers, however TensorFlow only provided image processing functions and layers. 
This led me to the creation of layers for a video processing pipeline.

Most of the implemented functions are analogous to their image counterparts from TensorFlow and Keras. 

So far, these are implemeted:
- [x] Random zoom (VideoRandomZoom)
- [x] Random rotation (videoRandomRotation)
- [x] Random crop (VideoRandomCrop)
- [x] Random contrast (VideoRandomContrast)
- [x] Random flip (VideoRandomFlip)
- [ ] Resizing
- [ ] Center crop
- [ ] Rescaling
- [ ] Random translation
- [ ] Random height
- [ ] Random width

## Usage

Include both using Sequential and Functional TF API, as you would normally do with image preprocessing layers

```python
import tensorflow as tf
from tf_video import VideoRandomFlip, VideoRandomContrast

input_video = ...
model = tf.keras.models.Sequential([VideoRandomFlip('horizontal_and_vertical'), VideoRandomContrast(0.3)])

aug = model(input_video)
```
