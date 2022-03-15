import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import glob
import math
import os

def train_test_split(imgs, masks):
    np.random.seed(127)
    train = np.random.choice(np.array(range(0, len(imgs))), math.ceil(.8*len(imgs)), replace=False)

    imgs_arr = np.array(imgs)
    train = list(imgs_arr[train])
    test = list(imgs_arr[[i for i in range(0, len(imgs)) if i not in train]])

    masks_arr = np.array(masks)
    train_masks = list(masks_arr[train])
    test_masks = list(masks_arr[[i for i in range(0, len(masks_arr)) if i not in train]])

    return train, train_masks, test, test_masks

def main(densenet=True):
    #train_test_split(imgs, masks)

    masks = sorted(glob.glob('data/masks/*'))
    imgs = sorted(glob.glob('data/train/*'))
    train_X = []
    train_Y = []
    for i in range(len(masks)):
        mask = cv2.imread(masks[i],cv2.IMREAD_GRAYSCALE)
        mask = mask.reshape(256,256,1)
        # only take patches which are a 1
        if np.sum(mask) > 0:
            train_Y.append(mask)
            img = cv2.imread(imgs[i],cv2.IMREAD_GRAYSCALE)
            img = img/255
            if densenet:
                img = -1*(img - np.max(img))
            img = img.reshape(256,256,1)
            train_X.append(img)

    masks = sorted(glob.glob('data/test_masks/*'))
    imgs = sorted(glob.glob('data/test/*'))
    test_X = []
    test_Y = []
    for i in range(len(masks)):
        mask = cv2.imread(masks[i],cv2.IMREAD_GRAYSCALE)
        mask = mask.reshape(256,256,1)
        # only take patches which are a 1
        if np.sum(mask) > 0:
            test_Y.append(mask)
            img = cv2.imread(imgs[i],cv2.IMREAD_GRAYSCALE)
            img = img/255
            if densenet:
                img = -1*(img - np.max(img))
            img = img.reshape(256,256,1)
            test_X.append(img)

    train_X = np.array(train_X, dtype=np.float32)
    train_Y = np.array(train_Y, dtype=np.float32)
    test_X = np.array(test_X, dtype=np.float32)
    test_Y = np.array(test_Y, dtype=np.float32)

    plt.figure(figsize=(12,6))
    plt.subplot(1, 2, 1)
    plt.imshow(test_X[14,...])
    plt.subplot(1, 2, 2)
    plt.imshow(test_Y[14,...,0])
    plt.savefig('figure/test_XY.png')

    print('train_X', train_X.shape)
    print('test_X', test_X.shape)

    torch.save({'train_X': train_X,
        'train_Y': train_Y,
        'test_X': test_X,
        'test_Y': test_Y,
        }, 'dataset.pt')

if __name__ == '__main__':
    main()
