import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def data_load(densenet=True):
    train = pd.read_csv('data/train_list.txt', header=None)
    imgs = ['data/train/%s' % (img) for img in train[0]]
    masks = ['data/masks/%s' % (img) for img in train[0]]
    train_X = []
    train_Y = []
    for i in range(len(masks)):
        mask = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
        mask = mask.reshape(256, 256, 1)
        # only take patches which are a 1
        if np.sum(mask) > 0:
            train_Y.append(mask)
            img = cv2.imread(imgs[i], cv2.IMREAD_GRAYSCALE)
            img = img / 255
            if densenet:
                img = -1 * (img - np.max(img))
            img = img.reshape(256, 256, 1)
            train_X.append(img)

    test = pd.read_csv('data/test_list.txt', header=None)
    imgs = ['data/test/%s' % (img) for img in test[0]]
    masks = ['data/test_masks/%s' % (img) for img in test[0]]
    test_X = []
    test_Y = []
    for i in range(len(masks)):
        mask = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
        mask = mask.reshape(256, 256, 1)
        # only take patches which are a 1
        if np.sum(mask) > 0:
            test_Y.append(mask)
            img = cv2.imread(imgs[i], cv2.IMREAD_GRAYSCALE)
            img = img / 255
            if densenet:
                img = -1 * (img - np.max(img))
            img = img.reshape(256, 256, 1)
            test_X.append(img)

    train_X = np.array(train_X, dtype=np.float32)
    train_Y = np.array(train_Y, dtype=np.float32)
    test_X = np.array(test_X, dtype=np.float32)
    test_Y = np.array(test_Y, dtype=np.float32)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(test_X[14, ...])
    plt.subplot(1, 2, 2)
    plt.imshow(test_Y[14, ..., 0])
    plt.savefig('figure/test_XY.png')

    print('train_X', train_X.shape)
    print('test_X', test_X.shape)

    return train_X, train_Y, test_X, test_Y

if __name__ == '__main__':
    data_load()
