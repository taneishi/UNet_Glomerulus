import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class KidneyDataset(Dataset):
    def __init__(self, phase, densenet=False, transform=None):
        df = pd.read_csv('data/%s_list.txt' % (phase), header=None)
        self.image_names = ['data/imgs/%s' % (image_name) for image_name in df[0]]
        self.mask_names = ['data/masks/%s' % (mask_name) for mask_name in df[0]]

        self.images, self.masks = [], []
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

class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        # Generate some random images
        self.input_images, self.target_masks = self.generate_random_data(192, 192, count=count)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, index):
        image = self.input_images[index]
        mask = self.target_masks[index]
        if self.transform:
            image = self.transform(image)

        return image, mask

    def generate_random_data(self, height, width, count):
        x, y = zip(*[self.generate_img_and_mask(height, width) for i in range(0, count)])

        X = np.asarray(x) * 255
        X = X.repeat(3, axis=1).transpose([0, 2, 3, 1]).astype(np.uint8)
        Y = np.asarray(y)

        return X, Y

    def get_random_location(self, width, height, zoom=1.0):
        x = int(width * random.uniform(0.1, 0.9))
        y = int(height * random.uniform(0.1, 0.9))

        size = int(min(width, height) * random.uniform(0.06, 0.12) * zoom)

        return x, y, size

    def logical_and(self, arrays):
        new_array = np.ones(arrays[0].shape, dtype=bool)
        for a in arrays:
            new_array = np.logical_and(new_array, a)

        return new_array

    def add_square(self, arr, x, y, size):
        s = int(size / 2)
        arr[x-s,y-s:y+s] = True
        arr[x+s,y-s:y+s] = True
        arr[x-s:x+s,y-s] = True
        arr[x-s:x+s,y+s] = True

        return arr

    def add_filled_square(self, arr, x, y, size):
        s = int(size / 2)

        xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

        return np.logical_or(arr, self.logical_and([xx > x - s, xx < x + s, yy > y - s, yy < y + s]))

    def add_mesh_square(self, arr, x, y, size):
        s = int(size / 2)

        xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

        return np.logical_or(arr, self.logical_and([xx > x - s, xx < x + s, xx % 2 == 1, yy > y - s, yy < y + s, yy % 2 == 1]))

    def add_triangle(self, arr, x, y, size):
        s = int(size / 2)

        triangle = np.tril(np.ones((size, size), dtype=bool))

        arr[x-s:x-s+triangle.shape[0], y-s:y-s+triangle.shape[1]] = triangle

        return arr

    def add_circle(self, arr, x, y, size, fill=False):
        xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]
        circle = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        new_arr = np.logical_or(arr, np.logical_and(circle < size, circle >= size * 0.7 if not fill else True))

        return new_arr

    def add_plus(self, arr, x, y, size):
        s = int(size / 2)
        arr[x-1:x+1, y-s:y+s] = True
        arr[x-s:x+s, y-1:y+1] = True

        return arr

    def generate_img_and_mask(self, height, width):
        shape = (height, width)

        triangle_location = self.get_random_location(*shape)
        circle_location1 = self.get_random_location(*shape, zoom=0.7)
        circle_location2 = self.get_random_location(*shape, zoom=0.5)
        mesh_location = self.get_random_location(*shape)
        square_location = self.get_random_location(*shape, zoom=0.8)
        plus_location = self.get_random_location(*shape, zoom=1.2)

        # Create input image
        arr = np.zeros(shape, dtype=bool)
        arr = self.add_triangle(arr, *triangle_location)
        arr = self.add_circle(arr, *circle_location1)
        arr = self.add_circle(arr, *circle_location2, fill=True)
        arr = self.add_mesh_square(arr, *mesh_location)
        arr = self.add_filled_square(arr, *square_location)
        arr = self.add_plus(arr, *plus_location)
        arr = np.reshape(arr, (1, height, width)).astype(np.float32)

        # Create target masks
        masks = np.asarray([
            self.add_filled_square(np.zeros(shape, dtype=bool), *square_location),
            self.add_circle(np.zeros(shape, dtype=bool), *circle_location2, fill=True),
            self.add_triangle(np.zeros(shape, dtype=bool), *triangle_location),
            self.add_circle(np.zeros(shape, dtype=bool), *circle_location1),
            self.add_filled_square(np.zeros(shape, dtype=bool), *mesh_location),
            #self.add_mesh_square(np.zeros(shape, dtype=bool), *mesh_location),
            self.add_plus(np.zeros(shape, dtype=bool), *plus_location)
        ]).astype(np.float32)

        return arr, masks

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
