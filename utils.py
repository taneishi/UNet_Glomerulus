import tensorflow as tf
from keras import backend as K

def dice_loss(smooth=1e-5):
    def dice(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred, smooth) 
    return dice

def dice_coef(y_true, y_pred, smooth=1):
    y_true = tf.clip_by_value(y_true, clip_value_min=0, clip_value_max=1)
    y_pred = tf.clip_by_value(y_pred, clip_value_min=0, clip_value_max=1)
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_inverse(y_true, y_pred, smooth=1):
    y_true = tf.clip_by_value(y_true, clip_value_min=0, clip_value_max=1)
    y_pred = tf.clip_by_value(y_pred, clip_value_min=0, clip_value_max=1)
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    
    return 1 - K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
