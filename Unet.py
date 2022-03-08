import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from keras.callbacks import EarlyStopping

def Unet(pretrained_weights=None, lr=.01, input_shape=(256,256,1)):    
    inputs = Input(input_shape) # input has size 256x256x1
    print('input:', inputs.shape)
    
    ### encoder

    # depth 0
    # There are 2 convolution layers
    # Number of filters=16, Kernel size=3, used relu activation function     
    conv1 = Conv2D(16, kernel_size =3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)   # 256x256x16 
    conv1 = Conv2D(16, kernel_size =3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)     
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 256x256x16 -> 128x128x16    
    pool1 = Dropout(0.1)(pool1) # dropout 10 percent 
    print('conv1: ',conv1.shape) 
    print('pool1: ',pool1.shape)
    
    # depth 1
    # There are 2 convolution layers
    # Number of filters=32, Kernel size=3  
    conv2 = Conv2D(32, kernel_size =3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)  # 128x128x32     
    conv2 = Conv2D(32, kernel_size =3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 128x128x32 -> 64x64x32    
    pool2 = Dropout(0.1)(pool2) # dropout 10 percent 
    print('conv2: ',conv2.shape) 
    print('pool2: ',pool2.shape)
    
    # depth 2    
    # There are 2 convolution layers
    # Number of filters=64, Kernel size=3  
    conv3 = Conv2D(64, kernel_size =3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)  # 64x64x64
    conv3 = Conv2D(64, kernel_size =3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 64x64x64 -> 32x32x64    
    pool3 = Dropout(0.1)(pool3) # dropout 10 percent 
    print('conv3: ',conv3.shape) 
    print('pool3: ',pool3.shape)

    # depth 3    
    # There are 2 convolution layers
    # Number of filters=128, Kernel size=3  
    conv4 = Conv2D(128, kernel_size =3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3) # 32x32x128
    conv4 = Conv2D(128, kernel_size =3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 32x32x128 -> 16x16x128    
    pool4 = Dropout(0.1)(pool4) # dropout 10 percent 
    print('conv4: ',conv4.shape) 
    print('pool4: ',pool4.shape)

    # depth 4 (choke)    
    # There are 2 convolution layers
    # Number of filters=256, Kernel size=3
    conv5 = Conv2D(256, kernel_size =3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)  # 16x16x256
    conv5 = Conv2D(256, kernel_size =3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    print('conv5: ',conv5.shape)   

    ### decoder    
        
    # Expansive path    
    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv4, up6], axis = 3)
    merge6=Dropout(0.1)(merge6) # dropout 10 percent 
    # depth 3
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)    
    print('up6:', up6.shape)   
    print('conv6:', conv6.shape) 
    
    up7 = Conv2D(64, 2,  activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3, up7], axis = 3)
    merge7=Dropout(0.1)(merge7) # dropout 10 percent 
    # depth 2
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)    
    print('up7:', up7.shape)
    print('conv7:', conv7.shape) 
    
    up8 = Conv2D(32,(3, 3),  activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2, up8], axis = 3)
    merge8=Dropout(0.1)(merge8) # dropout 10 percent 
    # depth 1
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)    
    print('up8:', up8.shape)
    print('conv8', conv8.shape)
    
    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1, up9], axis = 3)
    merge9=Dropout(0.1)(merge9) # dropout 10 percent 
    # depth 0   
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)   
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)    
    print('up9:', up9.shape)
    print('conv9', conv9.shape) 

    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    print('conv10:', conv10.shape)    
    
    model = Model(inputs = inputs, outputs = conv10)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=[dice_coef])
    if pretrained_weights != None:
        model.set_weights(pretrained_weights)    
    
    return model

#model = Unet(lr=.05)
model = Unet(lr=.00005)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_dice_coef_inverse', patience=3)
model.fit(train_X, train_Y, epochs=100, batch_size=10, verbose=1, validation_data=(test_X,test_Y), callbacks=[callback])

fit = model.predict(test_X)

for i in range(0,len(fit)):
    plt.imshow(fit[i,...,0])
    plt.pause(.2)
    plt.imshow(test_X[i,...,0])
    plt.pause(.2)
    plt.imshow(test_Y[i,...,0])
    plt.pause(.2)
