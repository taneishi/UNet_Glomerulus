import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import math
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
print(tf.config.list_physical_devices('GPU'))

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.callbacks import History 
from keras.callbacks import CSVLogger, EarlyStopping
from keras import activations

def dice_loss(smooth=1e-5):
    def dice(y_true, y_pred):
        return 1-dice_coef(y_true, y_pred, smooth) 
    return dice

def dice_coef(y_true, y_pred, smooth=1):
    y_true = tf.clip_by_value(y_true, clip_value_min=0, clip_value_max=1)
    y_pred = tf.clip_by_value(y_pred, clip_value_min=0, clip_value_max=1)
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_inverse(y_true, y_pred, smooth=1):
    y_true = tf.clip_by_value(y_true, clip_value_min=0, clip_value_max=1)
    y_pred = tf.clip_by_value(y_pred, clip_value_min=0, clip_value_max=1)
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    
    return 1-K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dense_block(inputs, filter_size):
    filter_size_4 = np.int16(filter_size/4)
    conv1 = Conv2D(filter_size_4, 3, padding = 'same',activation = 'relu', kernel_initializer = 'he_normal')(inputs)
    
    conv2 = Conv2D(filter_size_4, 3, padding = 'same',activation = 'relu', kernel_initializer = 'he_normal')(
        concatenate([inputs,conv1]))
    conv3 = Conv2D(filter_size_4, 3, padding = 'same',activation = 'relu', kernel_initializer = 'he_normal')(
        concatenate([inputs,conv1,conv2]))
    
    output = Conv2D(filter_size, 3, padding = 'same',activation = 'relu', kernel_initializer = 'he_normal')(
        concatenate([inputs,conv1,conv2,conv3]))
    
    return output

def get_down_block(inputs, filter_size, k = (3,3),choke = False):
    #feature_map = Conv2D(filter_size, 3, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    feature_map = dense_block(inputs,filter_size )

    if choke:
        return feature_map
    else:
        pool = MaxPooling2D(pool_size=(2,2), 
                            strides = (2,2),padding = 'same')(feature_map)
        return feature_map,pool

def get_up_block(inputs,merge,filter_size, k = (3,3)):
    up = Conv2DTranspose(filter_size, (2,2),strides = (2,2),activation = 'relu', padding = 'same',
                           kernel_initializer = 'he_normal')(inputs)
    merge = concatenate([up,merge],axis = 3)
    
    #feature_map = Conv2D(filter_size, 3, padding='same', kernel_initializer='he_normal', activation='relu')(merge)
    feature_map = dense_block(merge,filter_size )
    return feature_map

def res_unet(input_shape = (256,256,1),lr = .0001, depth_before_choke = 4):
    inputs = Input(input_shape)
    filters = [32]
    for i in range(0,depth_before_choke):
        filters.append(filters[i]*2)
            
    blocks = []
    # first block
    block,pool = get_down_block(inputs,filters[0])
    blocks.append(block)

    for i in range(1,depth_before_choke):
        block,pool = get_down_block(pool, filters[i])
        blocks.append(block)
        
    # choke block
    block = get_down_block(pool, filters[len(filters)-1], choke = True)
    
    # upsampling 
    filters = np.flip(filters)
    blocks = np.flip(blocks)
    for i in range(1,depth_before_choke+1):
        block = get_up_block(block,blocks[i-1],filters[i])

    output = Conv2D(1, 1, activation = 'relu', padding = 'same')(block)
    model = Model(inputs=inputs, outputs=output)
    model_dice=dice_loss(smooth=1e-5)
    model.compile(optimizer = Adam(lr = lr), loss = model_dice, metrics=[dice_coef,dice_coef_inverse])    
    return model
        
model = res_unet(lr=.00005)
history = History()
csv_logger = CSVLogger('training_res_unet.csv', append = True)
checkpoint = ModelCheckpoint('best_res_unet.hdf5', monitor='val_dice_coef_inverse', verbose=1,
    save_best_only=True, mode='auto', period=1)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_dice_coef_inverse', patience=3)
model.fit(train_X,train_Y, epochs = 100, batch_size = 10,verbose = 1,
        validation_data = (test_X,test_Y),
          callbacks=[callback,checkpoint,history,csv_logger])

fit = model.predict(test_X)

for i in range(0,len(fit)):
    plt.imshow(fit[i,...,0])
    plt.pause(.2)
    plt.imshow(test_X[i,...,0])
    plt.pause(.2)
    plt.imshow(test_Y[i,...,0])
    plt.pause(.2)
