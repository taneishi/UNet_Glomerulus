import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ReLU, Dropout, concatenate, add
from tensorflow.keras.optimizers import Adam

from preprocess import data_load
from utils import dice_coef

def residual_block(inputs,num):
    co1 = Conv2D(num, 3, padding='same', kernel_initializer='he_normal')(inputs)
    co1 = ReLU()(co1)
    co2 = Conv2D(num, 3, padding='same', kernel_initializer='he_normal')(co1)
    output = add([co1, co2])
    output = ReLU()(output)
    return output

def vanilla_unet(pretrained_weights=None, lr=.01, input_size=(256,256,1)):
    inputs = Input(input_size)
    print('input:', inputs.shape)
    # depth 0
    conv1 = residual_block(inputs, 32)
    conv1 = residual_block(conv1, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print('conv1: ', conv1.shape) 
    print('pool1: ', pool1.shape)
    # depth 1
    conv2 = residual_block(pool1, 64)
    conv2 = residual_block(conv2, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print('conv2: ', conv2.shape) 
    print('pool2: ', pool2.shape)
    # depth 2
    conv3 = residual_block(pool2, 128)
    conv3 = residual_block(conv3, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print('conv3: ', conv3.shape) 
    print('pool3: ', pool3.shape)
    # depth 3
    conv4 = residual_block(pool3, 256)
    conv4 = residual_block(conv4, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    print('conv4: ', conv4.shape) 
    print('pool4: ', pool4.shape)
    # depth 4
    conv5 = residual_block(pool4, 512)
    conv5 = residual_block(conv5, 512)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    print('conv5: ', conv5.shape) 
    print('pool5: ', pool5.shape)
    # depth 5 (choke)
    conv6 = residual_block(pool5, 1024)
    conv6 = residual_block(conv6, 1024)
    print('conv6: ', conv6.shape) 
        
    up7 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv6))
    merge7 = concatenate([conv5,up7], axis=3)
     
    # depth 4
    conv7 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    print('up7:', up7.shape)
    print('conv7', conv7.shape)
    
    up8 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv7))
    merge8 = concatenate([conv4,up8], axis=3)
    
    # depth 3
    conv8 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    print('up8:', up8.shape)
    print('conv8', conv8.shape)
    
    up9 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv8))
    merge9 = concatenate([conv3,up9], axis=3)
    
    # depth 2
    conv9 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    print('up9:', up9.shape)
    print('conv9', conv9.shape)
    
    up10 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv9))
    merge10 = concatenate([conv2,up10], axis=3)
    
    # depth 1
    conv10 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10)
    conv10 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    print('up10:', up10.shape)
    print('conv10', conv10.shape)
    
    up11 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv10))
    merge11 = concatenate([conv1,up11], axis=3)
    
    # depth 0
    conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge11)
    conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    conv10 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    conv11 = Conv2D(1, 1, activation='sigmoid')(conv10)
    print('up11:', up11.shape)
    print('conv11', conv11.shape)
    
    model = Model(inputs=inputs, outputs=conv11)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=[dice_coef])

    if pretrained_weights != None:
        model.set_weights(pretrained_weights)

    return model

train_X, train_Y = data_load('train_list.txt')
test_X, test_Y = data_load('test_list.txt')

model = vanilla_unet(lr=.00005)
model.fit(train_X, train_Y, epochs=100, batch_size=10, verbose=1, validation_data=(test_X, test_Y))

fit = model.predict(test_X)

for i in range(0,len(fit)):
    plt.imshow(fit[i,...,0])
    plt.savefig('figure/UnetResnet1.png')
    plt.imshow(test_X[i,...,0])
    plt.savefig('figure/UnetResnet2.png')
    plt.imshow(test_Y[i,...,0])
    plt.savefig('figure/UnetResnet3.png')
