import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def data_load(filename, densenet=True):
    df = pd.read_csv('data/%s' % (filename),  header=None)
    imgs = ['data/imgs/%s' % (img) for img in df[0]]
    masks = ['data/masks/%s' % (img) for img in df[0]]
    X, Y = [], []
    for i in range(len(masks)):
        mask = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
        mask = mask.reshape(256, 256, 1)
        # only take patches which are a 1
        if np.sum(mask) > 0:
            Y.append(mask)
            img = cv2.imread(imgs[i], cv2.IMREAD_GRAYSCALE)
            img = img / 255
            if densenet:
                img = -1 * (img - np.max(img))
            img = img.reshape(256, 256, 1)
            X.append(img)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    return X, Y

if __name__ == '__main__':
    train_X, train_Y = data_load('train_list.txt')
    print('train_X', train_X.shape)

    test_X, test_Y = data_load('test_list.txt')
    print('test_X', test_X.shape)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(test_X[14, ...])
    plt.subplot(1, 2, 2)
    plt.imshow(test_Y[14, ..., 0])
    plt.savefig('figure/test_XY.png')
