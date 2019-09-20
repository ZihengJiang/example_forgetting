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


# Introduce Gaussian noise to noise_percentage of image pixels
def noisy(image, noise_percentage, noise_std):
    row, col, ch = image.shape
    num_corrupt = int(np.floor(noise_percentage * row * col / 100))

    # Randomly choose pixels to add noise to
    xy_coords = np.random.choice(row * col, num_corrupt, replace=False)
    chan_coords = np.random.choice(ch, num_corrupt, replace=True)
    xy_coords = np.unravel_index(xy_coords, (row, col))

    out = np.copy(image)

    mean = 120

    # Add randomly generated Gaussian noise to pixels
    for coord in range(num_corrupt):
        noise = np.random.normal(mean, noise_std, 1)
        out[xy_coords[0][coord], xy_coords[1][coord],
            chan_coords[coord]] += noise

    return out


def noise_pixels(noise_percent_pixels, noise_std_pixels):
    # Introduce noise to images if specified
    for ind in range(len(train_indx)):
        image = train_ds.data[ind, :, :, :]
        noisy_image = noisy(image, args.noise_percent_pixels, args.noise_std_pixels)
        train_ds.data[ind, :, :, :] = noisy_image


def noise_labels(dataset, noise_percent_labels, fname):
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
        nlabels = len(train_ds)
        nlabels_to_change = int(noise_percent_labels * nlabels / 100)
        nclasses = len(np.unique(labels))
        print('flipping ' + str(nlabels_to_change) + ' labels')

        # Randomly choose which labels to change, get indices
        change_indexes = npr.choice(
            np.arange(nlabels), nlabels_to_change, replace=False)

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
    if name in ['cifar10', 'cifar100']:
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
