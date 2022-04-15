import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class KidneyDataset(Dataset):
    def __init__(self, phase, densenet=False, transform=None):
        df = pd.read_csv('data/%s_list.txt' % (phase), header=None)
        self.image_names = ['data/imgs/%s' % (image_name) for image_name in df[0]]
        self.mask_names = ['data/masks/%s' % (mask_name) for mask_name in df[0]]

        self.images = []
        self.masks = []

        for image_name, mask_name in zip(self.image_names, self.mask_names):
            mask = Image.open(mask_name).convert('L') # convert to grayscale

            if np.sum(mask) > 0: # only take patches which are a 1
                self.masks.append(mask)
                image = Image.open(image_name).convert('RGB')
                if densenet:
                    image = -1 * (image - np.max(image))
                self.images.append(image)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        ])

    train_set = KidneyDataset('train', transform=transform)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)

    test_set = KidneyDataset('test', transform=transform)
    test_loader = DataLoader(test_set, batch_size=5, shuffle=False, num_workers=0)

    print('train set', len(train_set), 'test set', len(test_set))

    for images, masks in test_loader:
        print(images.shape, masks.shape)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(images[4].numpy().transpose(1, 2, 0))
        plt.subplot(1, 2, 2)
        plt.imshow(masks[4].numpy().transpose(1, 2, 0))
        plt.tight_layout()
        plt.savefig('figure/input.png')

        break
