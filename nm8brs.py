# -*- coding: utf-8 -*-
'''VGG16 model for Keras.
# Reference:
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
'''
from __future__ import print_function
from __future__ import absolute_import

import warnings
import os

from keras.models import Model
from keras.layers import Flatten, Dense, Input,Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions, preprocess_input


TF_WEIGHTS_PATH = os.path.join("models","nm8rs.h5")
TF_WEIGHTS_NO_TOP_PATH = os.path.join("models","nm8rs_no_top.h5")


def NM8RS(include_top=False,input_tensor=None):   

    
    if include_top:
        input_shape =(28,28,1)
    else:
        input_shape = (None,None,1)
        
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    

    x = Conv2D(16,3,3, activation='relu',border_mode="same",name='block1_conv')(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(64,3,3, activation='relu',border_mode="same",name='block2_conv')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Conv2D(128,3,3, activation='relu',border_mode="same",name='block3_conv')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    if include_top:
        x = Flatten(name='flatten')(x)
        x = Dense(256, activation='relu',name='fc1')(x)

        x = Dense(10, activation='softmax',name='predictions')(x)
        
    model = Model(input=img_input, output=x)
    
    if include_top:
        model.load_weights(TF_WEIGHTS_PATH)
    else:
        model.load_weights(TF_WEIGHTS_NO_TOP_PATH)

    return model