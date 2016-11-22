# -*- coding: utf-8 -*-
'''VGG16 model for Keras.
# Reference:
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
'''
from __future__ import print_function
from __future__ import absolute_import

import warnings
import os

from ..models import Model
from ..layers import Flatten, Dense, Input
from ..layers import Convolution2D, MaxPooling2D
from ..utils.layer_utils import convert_all_kernels_in_model
from ..utils.data_utils import get_file
from .. import backend as K
from .imagenet_utils import decode_predictions, preprocess_input


TF_WEIGHTS_PATH = os.path.join("models","nm8rs.h5")


def NM8RS(input_tensor=None):
    '''Instantiate the VGG16 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (1, 28, 28)
        else:
            input_shape = (1, None, None)
    else:
        if include_top:
            input_shape = (28, 28, 1)
        else:
            input_shape = (None, None, 1)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
            
    inputs = Input(shape=(28,28,1))

    x = Conv2D(16,3,3, activation='relu',border_mode="same",name='block1_conv')(inputs)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(64,3,3, activation='relu',border_mode="same",name='block2_conv')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Conv2D(128,3,3, activation='relu',border_mode="same",name='block3_conv')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu',name='fc1')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(10, activation='softmax',name='predictions')(x)

    model = Model(input=inputs, output=predictions)

    # load weights
    if weights == 'nm8brs':
        if K.image_dim_ordering() == 'th':
            model.load_weights(TF_WEIGHTS_PATH)
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(model)
        else:
            model.load_weights(TF_WEIGHTS_PATH)
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)
    return model