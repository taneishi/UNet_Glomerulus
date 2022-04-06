import numpy as np
import matplotlib.pyplot as plt

from preprocess import data_load

train_X, train_Y = data_load('train_list.txt')
test_X, test_Y = data_load('test_list.txt')

model = densenet(lr=.00005)

model_dice = dice_loss(smooth=1e-5)
model.compile(optimizer=Adam(lr=lr), loss=model_dice, metrics=[dice_coef, dice_coef_inverse])

model.fit(train_X, train_Y, epochs=100, batch_size=10, verbose=1, validation_data=(test_X, test_Y))

fit = model.predict(test_X)

for i in range(0, len(fit)):
    plt.imshow(fit[i, ..., 0])
    plt.savefig('figure/DenseNet1.png')
    plt.imshow(test_X[i, ..., 0])
    plt.savefig('figure/DenseNet2.png')
    plt.imshow(test_Y[i, ..., 0])
    plt.savefig('figure/DenseNet3.png')
