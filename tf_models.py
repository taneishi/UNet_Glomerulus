from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, ReLU, Dropout, concatenate, add
from tensorflow.keras.optimizers import Adam

def Unet(pretrained_weights=None, learning_rate=.01, input_shape=(256, 256, 1)):
    inputs = Input(input_shape) # input has size 256x256x1
    print('input:', inputs.shape)

    ### encoder

    # depth 0
    # There are 2 convolution layers
    # Number of filters=16, Kernel size=3, used relu activation function
    conv1 = Conv2D(16, kernel_size =3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs) # 256x256x16
    conv1 = Conv2D(16, kernel_size =3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # 256x256x16 -> 128x128x16
    pool1 = Dropout(0.1)(pool1) # dropout 10 percent
    print('conv1:', conv1.shape)
    print('pool1:', pool1.shape)

    # depth 1
    # There are 2 convolution layers
    # Number of filters=32, Kernel size=3
    conv2 = Conv2D(32, kernel_size =3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1) # 128x128x32
    conv2 = Conv2D(32, kernel_size =3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) # 128x128x32 -> 64x64x32
    pool2 = Dropout(0.1)(pool2) # dropout 10 percent
    print('conv2:', conv2.shape)
    print('pool2:', pool2.shape)

    # depth 2
    # There are 2 convolution layers
    # Number of filters=64, Kernel size=3
    conv3 = Conv2D(64, kernel_size =3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2) # 64x64x64
    conv3 = Conv2D(64, kernel_size =3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) # 64x64x64 -> 32x32x64
    pool3 = Dropout(0.1)(pool3) # dropout 10 percent
    print('conv3:', conv3.shape)
    print('pool3:', pool3.shape)

    # depth 3
    # There are 2 convolution layers
    # Number of filters=128, Kernel size=3
    conv4 = Conv2D(128, kernel_size =3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3) # 32x32x128
    conv4 = Conv2D(128, kernel_size =3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) # 32x32x128 -> 16x16x128
    pool4 = Dropout(0.1)(pool4) # dropout 10 percent
    print('conv4:', conv4.shape)
    print('pool4:', pool4.shape)

    # depth 4 (choke)
    # There are 2 convolution layers
    # Number of filters=256, Kernel size=3
    conv5 = Conv2D(256, kernel_size =3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4) # 16x16x256
    conv5 = Conv2D(256, kernel_size =3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    print('conv5:', conv5.shape)

    ### decoder

    # Expansive path
    up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3)
    merge6 = Dropout(0.1)(merge6) # dropout 10 percent
    # depth 3
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    print('up6:', up6.shape)
    print('conv6:', conv6.shape)

    up7 = Conv2D(64, 2,  activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    merge7 = Dropout(0.1)(merge7) # dropout 10 percent
    # depth 2
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    print('up7:', up7.shape)
    print('conv7:', conv7.shape)

    up8 = Conv2D(32,(3, 3),  activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    merge8 = Dropout(0.1)(merge8) # dropout 10 percent
    # depth 1
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    print('up8:', up8.shape)
    print('conv8', conv8.shape)

    up9 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    merge9 = Dropout(0.1)(merge9) # dropout 10 percent
    # depth 0
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    print('up9:', up9.shape)
    print('conv9', conv9.shape)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    print('conv10:', conv10.shape)

    model = Model(inputs=inputs, outputs=conv10)

    return model

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
    print('conv1:', conv1.shape)
    print('pool1:', pool1.shape)
    # depth 1
    conv2 = residual_block(pool1, 64)
    conv2 = residual_block(conv2, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print('conv2:', conv2.shape)
    print('pool2:', pool2.shape)
    # depth 2
    conv3 = residual_block(pool2, 128)
    conv3 = residual_block(conv3, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print('conv3:', conv3.shape)
    print('pool3:', pool3.shape)
    # depth 3
    conv4 = residual_block(pool3, 256)
    conv4 = residual_block(conv4, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    print('conv4:', conv4.shape)
    print('pool4:', pool4.shape)
    # depth 4
    conv5 = residual_block(pool4, 512)
    conv5 = residual_block(conv5, 512)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    print('conv5:', conv5.shape)
    print('pool5:', pool5.shape)
    # depth 5 (choke)
    conv6 = residual_block(pool5, 1024)
    conv6 = residual_block(conv6, 1024)
    print('conv6:', conv6.shape)

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

    return model

def dense_block(inputs, filter_size):
    filter_size_4 = np.int16(filter_size / 4)
    conv1 = Conv2D(filter_size_4, 3, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    conv2 = Conv2D(filter_size_4, 3, padding='same',activation='relu', kernel_initializer='he_normal')(concatenate([inputs, conv1]))
    conv3 = Conv2D(filter_size_4, 3, padding='same',activation='relu', kernel_initializer='he_normal')(concatenate([inputs, conv1, conv2]))
    output = Conv2D(filter_size, 3, padding='same', activation='relu', kernel_initializer='he_normal')(concatenate([inputs, conv1, conv2, conv3]))

    return output

def get_down_block(inputs, filter_size, k=(3, 3), choke=False):
    #feature_map = Conv2D(filter_size, 3, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    feature_map = dense_block(inputs,filter_size )

    if choke:
        return feature_map
    else:
        pool = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(feature_map)
        return feature_map,pool

def get_up_block(inputs,merge,filter_size, k=(3,3)):
    up = Conv2DTranspose(filter_size, (2,2), strides=(2,2), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    merge = concatenate([up, merge], axis=3)

    #feature_map = Conv2D(filter_size, 3, padding='same', kernel_initializer='he_normal', activation='relu')(merge)
    feature_map = dense_block(merge, filter_size)
    return feature_map

def densenet(input_shape=(256, 256, 1), lr=.0001, depth_before_choke=4):
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
    block = get_down_block(pool, filters[len(filters)-1], choke=True)

    # upsampling
    filters = np.flip(filters)
    blocks = np.flip(blocks)

    for i in range(1, depth_before_choke+1):
        block = get_up_block(block,blocks[i-1], filters[i])

    output = Conv2D(1, 1, activation='relu', padding='same')(block)
    model = Model(inputs=inputs, outputs=output)

    return model
