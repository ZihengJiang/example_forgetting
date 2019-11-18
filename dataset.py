import os
import pickle
import numpy as np
import numpy.random as npr

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)

class CIFAR10TFDS(Dataset):
    train_list = [
        'data_batch_1',
        'data_batch_2',
        'data_batch_3',
        'data_batch_4',
        'data_batch_5',
    ]
    test_list = ['test_batch']
    def __init__(self, root, train, transform=None):
        self.root = root
        self.train = train
        self.transform=transform

        if self.train:
            files = self.train_list
        else:
            files = self.test_list

        self.data = []
        self.targets = []
        for file_name in files:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f)
                print('entry')
                print(entry)
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
        print('data')
        print(self.data)
        raise ValueError
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

# Introduce Gaussian noise to noise_percentage of image pixels
def noisy(image, noise_percentage, noise_std):
    row, col, ch = image.shape
    num_corrupt = int(np.floor(noise_percentage * row * col / 100))

    # Randomly choose pixels to add noise to
    xy_coords = np.random.choice(row * col, num_corrupt, replace=False)
    chan_coords = np.random.choice(ch, num_corrupt, replace=True)
    xy_coords = np.unravel_index(xy_coords, (row, col))

    out = np.copy(image)

    # Add randomly generated Gaussian noise to pixels
    for coord in range(num_corrupt):
        if noise_std == -1.0:
            # replace with random pixel
            noise = np.random.randint(0, 255)
            out[xy_coords[0][coord], xy_coords[1][coord],
                chan_coords[coord]] = noise
        else:
            # add noise on pixels
            mean = 120
            noise = np.random.normal(mean, noise_std, 1)
            out[xy_coords[0][coord], xy_coords[1][coord],
                chan_coords[coord]] += noise

    return out


def noise_pixels(dataset, noise_percent, noise_pixels_percent, noise_pixels_std, fname):
    # Introduce noise to images if specified
    assert isinstance(dataset, IndexedDataset)
    train_ds = dataset.dataset
    if hasattr(train_ds, "labels"):
        labels = train_ds.labels
    elif hasattr(train_ds, "targets"):
        labels = train_ds.targets
    else:
        raise ValueError("check pytorch dataset api change")

    # Introduce noise to labels if specified
    with open(fname + "_changed_pixels.txt", "w") as f:
        # Compute number of labels to change
        num_data = len(train_ds)
        num_data_to_change = int(noise_percent * num_data / 100)
        nclasses = len(np.unique(labels))
        print('add noise on ' + str(num_data_to_change) + ' pixels')

        # Randomly choose which labels to change, get indices
        change_indexes = npr.choice(
            np.arange(num_data), num_data_to_change, replace=False)

        for l, idx in enumerate(change_indexes):
            image = train_ds.data[idx, :, :, :]
            noisy_image = noisy(image, noise_pixels_percent, noise_pixels_std)
            train_ds.data[idx, :, :, :] = noisy_image
            f.write(str(idx)  + '\n')
    return IndexedDataset(train_ds), change_indexes



def noise_labels(dataset, noise_percent, fname):
    assert isinstance(dataset, IndexedDataset)
    train_ds = dataset.dataset
    if hasattr(train_ds, "labels"):
        labels = train_ds.labels
    elif hasattr(train_ds, "targets"):
        labels = train_ds.targets
    else:
        raise ValueError("check pytorch dataset api change")

    # Introduce noise to labels if specified
    with open(fname + "_changed_labels.txt", "w") as f:
        # Compute number of labels to change
        num_data = len(train_ds)
        num_data_to_change = int(noise_percent * num_data / 100)
        nclasses = len(np.unique(labels))
        print('flipping ' + str(nlabels_to_change) + ' labels')

        # Randomly choose which labels to change, get indices
        change_indexes = npr.choice(
            np.arange(num_data), num_data_to_change, replace=False)

        # Flip each of the randomly chosen labels
        for l, idx in enumerate(change_indexes):
            label_choices = np.arange(nclasses)
            true_label = labels[idx]
            label_choices = np.delete(
                label_choices,
                true_label)  # the label is the same as the index of the label

            # Get new label and relabel the example with it
            noisy_label = npr.choice(label_choices, 1)
            # train_ds.targets[idx] = torch.tensor(noisy_label[0])
            labels[idx] = noisy_label[0]

            # Write the example index from the original example order, the old, and the new label
            f.write(
                str(idx) + ' ' + str(true_label) +
                ' ' + str(noisy_label[0]) + '\n')
    return IndexedDataset(train_ds), change_indexes


def get_data(name):
    if name in ['cifar10', 'cifar100', 'cifar10tfds']:
        return get_cifar(name)
    elif name == 'svhn':
        return get_svhn()
    elif name == 'mnist':
        return get_mnist()
    else:
        raise ValueError


def get_mnist():
    # Setup transforms
    all_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ]
    transform = transforms.Compose(all_transforms)

    # Load the appropriate train and test datasets
    train_ds = datasets.MNIST(
        root='/tmp/data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(
        root='/tmp/data', train=False, download=True, transform=transform)
    num_classes = 10

    return IndexedDataset(train_ds), IndexedDataset(test_ds), num_classes


def get_cifar(name):
    # Image Preprocessing
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    # Setup train transforms
    train_transform = transforms.Compose([])
    # if args.data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    # Setup test transforms
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Load the appropriate train and test datasets
    if name == 'cifar10':
        num_classes = 10
        train_ds = datasets.CIFAR10(
            root='/tmp/data/',
            train=True,
            transform=train_transform,
            download=True)

        test_ds = datasets.CIFAR10(
            root='/tmp/data/',
            train=False,
            transform=test_transform,
            download=True)
    elif name == 'cifar100':
        num_classes = 100
        train_ds = datasets.CIFAR100(
            root='/tmp/data/',
            train=True,
            transform=train_transform,
            download=True)

        test_ds = datasets.CIFAR100(
            root='/tmp/data/',
            train=False,
            transform=test_transform,
            download=True)
    elif name == 'cifar10tfds':
        num_classes = 10
        train_ds = CIFAR10TFDS(
            root='/sampa/home/ziheng/downloads/cifar10-tfds-export',
            train=True,
            transform=train_transform)
        test_ds = CIFAR10TFDS(
            root='/sampa/home/ziheng/downloads/cifar10-tfds-export',
            train=False,
            transform=test_transform)
    else:
        raise ValueError
    return IndexedDataset(train_ds), IndexedDataset(test_ds), num_classes

def get_svhn():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_ds = datasets.SVHN(root='/tmp/data/', split='train', download=True, transform=transform)
    test_ds = datasets.SVHN(root='/tmp/data/', split='test', download=True, transform=transform)
    num_classes = 10
    print(dir(train_ds))
    return IndexedDataset(train_ds), IndexedDataset(test_ds), num_classes


