import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader


class Cutout(object):
    def __init__(self, length = 16, prob = 0.5):
        self.length = length
        self.prob = prob

    def __call__(self, img):
        if np.random.rand(1) < self.prob:

            h = img.size(1)
            w = img.size(2)

            mask = torch.ones(h, w)

            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

            mask = mask.expand_as(img)
            img = img * mask

        return img


def load(root = '../data/', train = True, batch_size = 128, shuffle = True, num_workers = 4,
         cutout = False, length = 16, prob = 0.5):
    torch.manual_seed(0)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])
    if cutout:
        transform.transforms.append(Cutout(length = length, prob = prob))

    dataset = datasets.CIFAR100(root = root, train = train, download = True, transform = transform)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)

    return loader


if __name__ == '__main__':
    train_loader = load(cutout = True, prob = 1)

    for inputs, targets in train_loader:
        print(inputs[0])
        print(targets[0])

        print(inputs[0].shape)
        print(inputs[0].max())
        print(inputs[0].min())

        for index in range(3):
            img = np.transpose(inputs[index], [1, 2, 0])
            plt.imshow(img * 0.5 + 0.5)
            plt.show()

        break
