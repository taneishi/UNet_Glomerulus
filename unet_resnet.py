import matplotlib.pyplot as plt

from preprocess import data_load
from utils import dice_coef

train_X, train_Y = data_load('train_list.txt')
test_X, test_Y = data_load('test_list.txt')

model = vanilla_unet(lr=.00005)

model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=[dice_coef])

if pretrained_weights != None:
    model.set_weights(pretrained_weights)

model.fit(train_X, train_Y, epochs=100, batch_size=10, verbose=1, validation_data=(test_X, test_Y))

fit = model.predict(test_X)

for i in range(0, len(fit)):
    plt.imshow(fit[i, ..., 0])
    plt.savefig('figure/UnetResnet1.png')
    plt.imshow(test_X[i, ..., 0])
    plt.savefig('figure/UnetResnet2.png')
    plt.imshow(test_Y[i, ..., 0])
    plt.savefig('figure/UnetResnet3.png')
