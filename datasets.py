import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2

class KidneyDataset(Dataset):
    def __init__(self, phase, densenet=False, transform=None):
        df = pd.read_csv('data/%s_list.txt' % (phase), header=None)
        self.image_names = ['data/imgs/%s' % (image_name) for image_name in df[0]]
        self.mask_names = ['data/masks/%s' % (mask_name) for mask_name in df[0]]

        self.images = []
        self.masks = []

        for image_name, mask_name in zip(self.image_names, self.mask_names):
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
            mask = mask.reshape(256, 256, 1)

            if np.sum(mask) > 0: # only take patches which are a 1
                self.masks.append(mask)
                image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
                image = image / 255
                if densenet:
                    image = -1 * (image - np.max(image))
                image = image.reshape(256, 256, 1)
                self.images.append(image)

        #X = np.array(X, dtype=np.float32)
        #Y = np.array(Y, dtype=np.float32)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        if self.transform:
            image = self.transform(image)

        return image, mask

if __name__ == '__main__':
    train_set = KidneyDataset('train')
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)

    test_set = KidneyDataset('test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    print('train set', len(train_set), 'test set', len(test_set))

    for image, mask in train_loader:
        print(image.shape, mask.shape)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image.reshape(256, 256, 1))
        plt.subplot(1, 2, 2)
        plt.imshow(mask.reshape(256, 256, 1))
        plt.savefig('figure/input.png')

        break
