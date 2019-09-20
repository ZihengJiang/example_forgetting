import pdb
import argparse
import numpy as np
import numpy.random as npr
import time
import os
import sys
import pickle

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from utils import *
from stats import *
from model.resnet import ResNet18
from model.wide_resnet import WideResNet
from model.cnn import CNN
from dataset import get_data, noise_labels



def eval_on_batch(model, criterion, inputs, targets, indexes, example_stats):
    # Forward propagation, compute loss, get predictions
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    _, predicted = torch.max(outputs.data, 1)

    # Update statistics and loss
    acc = predicted == targets
    for idx, index in enumerate(indexes):
        index_in_original_dataset = index.item()
        # Compute misclassification margin
        normalized_outputs = torch.nn.Softmax(dim=1)(outputs)
        output_correct_class = normalized_outputs.data[idx, targets[idx].item()]
        sorted_output, _ = torch.sort(normalized_outputs.data[idx, :])
        if acc[idx]:
            # Example classified correctly, highest incorrect class is 2nd largest output
            output_highest_incorrect_class = sorted_output[-2]
        else:
            # Example misclassified, highest incorrect class is max output
            output_highest_incorrect_class = sorted_output[-1]
        correct_confidence = output_correct_class.item()
        incorrect_confidence = output_highest_incorrect_class.item()
        # margin = output_correct_class.item() - output_highest_incorrect_class.item()

        # Add the statistics of the current training example to dictionary
        index_stats = example_stats.get(index_in_original_dataset,
                                        [[], [], []])
        index_stats[0].append(loss[idx].item())
        index_stats[1].append(acc[idx].sum().item())
        index_stats[2].append((correct_confidence, incorrect_confidence))
        example_stats[index_in_original_dataset] = index_stats

    return outputs, loss


# Train model for one epoch
#
# example_stats: dictionary containing statistics accumulated over every presentation of example
#
def train(args, model, criterion, device, loader, model_optimizer, epoch,
          example_stats):
    total = 0.
    moving_loss = 0.
    moving_correct = 0.

    model.train()


    print('\n=> Training Epoch #%d' % (epoch))
    for batch_idx, data in enumerate(loader, 0):
        model_optimizer.zero_grad()
        inputs, targets = data[0].to(device), data[1].to(device)
        example_indexes = data[2]

        outputs, loss = eval_on_batch(model, criterion, inputs, targets, example_indexes, example_stats)

        # Update loss, backward propagate, update optimizer
        total += targets.size(0)
        _, predicted = torch.max(outputs.data, 1)
        moving_correct += predicted.eq(targets.data).cpu().sum()

        loss = loss.mean()
        loss.backward()
        model_optimizer.step()
        moving_loss += loss.item()

        flush_print(
            '| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %
            (epoch, args.epochs, batch_idx + 1, len(loader),
             loss.item(), 100. * moving_correct.item() / total))

        # Add training accuracy to dict
        index_stats = example_stats.get('train', [[], []])
        index_stats[1].append(100. * moving_correct.item() / float(total))
        example_stats['train'] = index_stats

    # Save training accuracy for every epoch to dict
    index_stats = example_stats.get('train_acc', [[], []])
    index_stats[1].append(100. * moving_correct.item() / float(total))
    example_stats['train_acc'] = index_stats


def evaluate_on_removed(model, criterion, device, loader, epoch, example_stats):
    total = 0.
    moving_loss = 0.
    moving_correct = 0.

    model.eval()

    for batch_idx, data in enumerate(loader, 0):
        inputs, targets = data[0].to(device), data[1].to(device)
        example_indexes = data[2]

        outputs, loss = eval_on_batch(model, criterion, inputs, targets, example_indexes, example_stats)
        loss = loss.mean()
        total += targets.size(0)
        _, predicted = torch.max(outputs.data, 1)
        moving_correct += predicted.eq(targets.data).cpu().sum()
        acc = 100. * moving_correct.item() / total

    print("| Removed Set Epoch #%d\t\tLoss: %.4f Acc@1: %.2f%%" %
          (epoch, loss.item(), acc))
    index_stats = example_stats.get('removed_acc', [[], []])
    index_stats[1].append(100. * moving_correct.item() / float(total))
    example_stats['removed_acc'] = index_stats


# Evaluate model predictions on heldout test data
#
# example_stats: dictionary containing statistics accumulated over every presentation of example
#
best_acc = 0.
def test(epoch, model, criterion, device, loader, example_stats, checkpoint_fname):
    global best_acc
    test_loss = 0.
    correct = 0.
    total = 0.
    test_batch_size = 32

    model.eval()

    for batch_idx, data in enumerate(loader, 0):
        inputs, targets = data[0].to(device), data[1].to(device)
        # Forward propagation, compute loss, get predictions
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss = loss.mean()
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Add test accuracy to dict
    acc = 100. * correct.item() / total
    index_stats = example_stats.get('test', [[], []])
    index_stats[1].append(100. * correct.item() / float(total))
    example_stats['test'] = index_stats
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %
          (epoch, loss.item(), acc))

    # Save checkpoint when best model
    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' % (acc))
        state = {
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, checkpoint_fname)
        best_acc = acc


def main(args):
    # Enter all arguments that you want to be in the filename of the saved output
    ordered_args = [
        'dataset', 'data_augmentation', 'seed', 'sorting_file',
        'remove_percent', 'burn_in_epochs', 'remove_strategy',
        'noise_percent_labels',
    ]
    save_fname = '__'.join(
        '{}_{}'.format(arg, args_dict[arg]) for arg in ordered_args)
    fname = os.path.join(args.output_dir, save_fname)
    if os.path.exists(fname + '__stats_dict.pkl'):
        redo = input("There exists experiment result already, continue? [yes/no] ")
        if redo == 'no':
            exit()
        elif redo == 'yes':
            pass
        else:
            raise ValueError('wrong answer')


    os.makedirs(args.output_dir, exist_ok=True)

    # Set appropriate devices
    device = torch.device(args.device)
    print('run on device: {0}'.format(device))
    cudnn.benchmark = True  # Should make training go faster for large models

    # Set random seed for initialization
    torch.manual_seed(args.seed)
    if 'cuda' in args.device:
        torch.cuda.manual_seed(args.seed)
    npr.seed(args.seed)

    train_ds, test_ds, num_classes = get_data(args.dataset)

    if args.noise_percent_pixels:
        pass

    if args.noise_percent_labels:
        train_ds, noise_indexes = noise_labels(train_ds, args.noise_percent_labels, fname)

    print('Training on ' + str(len(train_ds)) + ' examples')

    # Setup model
    if args.model == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif args.model == 'wideresnet':
        if args.dataset == 'svhn':
            model = WideResNet(
                depth=16, num_classes=num_classes, widen_factor=8, dropRate=0.4)
        else:
            model = WideResNet(
                depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)
    elif args.model == 'cnn':
        model = CNN(num_classes=num_classes)
    else:
        print(
            'Specified model not recognized. Options are: resnet18 and wideresnet')

    # Setup loss
    model = model.to(args.device)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    criterion.__init__(reduce=False)

    # Setup optimizer
    if args.optimizer == 'adam':
        model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif args.optimizer == 'sgd':
        model_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4)
        scheduler = MultiStepLR(
            model_optimizer, milestones=[60, 120, 160], gamma=0.2)
    else:
        print('Specified optimizer not recognized. Options are: adam and sgd')

    save_point = os.path.join(args.output_dir, 'checkpoint', args.dataset)
    os.makedirs(save_point, exist_ok=True)
    checkpoint_fname = os.path.join(save_point, save_fname + '.t7')

    # Initialize dictionary to save statistics for every example presentation
    example_stats = {}
    num_examples = len(train_ds)
    example_weights = np.ones(num_examples)

    elapsed_time = 0
    # train_idx = np.array(range(0, len(train_ds)))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        if args.remove_strategy != 'normal' and epoch >= args.burn_in_epochs:
            if 'sampling' in args.remove_strategy:
                # sampling by weight
                normalized_weights = example_weights / example_weights.sum()
                index_stats = example_stats.get('example_weights', [[], []])
                index_stats[1].append(normalized_weights)
                example_stats['example_weights'] = index_stats

                choice_num = int(num_examples * (1 - args.remove_percent / 100))
                train_idx = np.random.choice(range(num_examples), size=choice_num,
                    replace=False, p=normalized_weights)
            elif args.remove_strategy == 'low-acc':
                remove_n = int(args.remove_percent * num_examples / 100)
                losses = []
                for idx in range(num_examples):
                    losses.append(example_stats[idx][0][epoch-1])
                losses = np.array(losses)
                sorted_indexes = np.argsort(losses)
                train_idx = sorted_indexes[:num_examples - remove_n]
            elif args.remove_strategy == 'all-noise':
                remove_n = int(args.remove_percent * num_examples / 100)
                if args.remove_percent <= args.noise_percent_labels:
                    remove_indexes = npr.choice(noise_indexes, remove_n, replace=False)
                    train_idx = np.setdiff1d(range(num_examples), remove_indexes)
                else:
                    train_idx = np.setdiff1d(range(num_examples), noise_indexes)
                    train_idx = npr.choice(train_idx, num_examples - remove_n, replace=False)
            else:
                # event method
                _, unlearned_per_presentation, _, first_learned = compute_forgetting_statistics(example_stats, epoch)
                ordered_examples, ordered_values = sort_examples_by_forgetting([unlearned_per_presentation], [first_learned], epoch)
                train_idx = sample_dataset_by_forgetting(train_ds, ordered_examples, ordered_values, args.remove_percent, args.remove_strategy)
            sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)

        start_time = time.time()
        train(args, model, criterion, device, train_loader, model_optimizer, epoch,
              example_stats)

        test_loader = DataLoader(test_ds, batch_size=32, shuffle=True)
        test(epoch, model, criterion, device, test_loader, example_stats, checkpoint_fname)

        if args.remove_strategy != 'normal' and epoch >= args.burn_in_epochs:
            # evaluate on removed data
            removed_idx =  np.setdiff1d(range(num_examples), train_idx)
            sampler = torch.utils.data.SubsetRandomSampler(removed_idx)
            removed_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
            evaluate_on_removed(model, criterion, device, removed_loader, epoch, example_stats)

        if 'sampling' in args.remove_strategy:
            example_weights = update_example_weights(example_weights, example_stats, epoch, args.remove_strategy)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

        # Update optimizer step
        if args.optimizer == 'sgd':
            scheduler.step(epoch)

        # Save the stats dictionary
        fname = os.path.join(args.output_dir, save_fname)
        with open(fname + "__stats_dict.pkl", "wb") as f:
            pickle.dump(example_stats, f)

        # Log the best train and test accuracy so far
        with open(fname + "__best_acc.txt", "w") as f:
            f.write('train test \n')
            f.write(str(max(example_stats['train'][1])))
            f.write(' ')
            f.write(str(max(example_stats['test'][1])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10', 'cifar100', 'svhn'])
    parser.add_argument('--model', default='resnet18', choices=['cnn', 'resnet18', 'wideresnet', ])
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='input batch size for training (default: 128)')
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='number of epochs to train (default: 200)')
    parser.add_argument(
        '--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument(
        '--data_augmentation',
        action='store_true',
        default=False,
        help='augment data by flipping and cropping')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--sorting_file',
        default="none",
        help=
        'name of a file containing order of examples sorted by forgetting (default: "none", i.e. not sorted)'
    )
    parser.add_argument(
        '--remove_percent',
        type=int,
        default=0,
        help='number of sorted examples to remove from training')
    parser.add_argument(
        '--remove_strategy',
        type=str,
        default='unforgettable',
        choices=['normal', 'random', 'forgettable', 'unforgettable',  'low-acc', 'all-noise', 'sampling', 'sampling1', 'sampling2', 'sampling3'])
    parser.add_argument(
        '--burn_in_epochs',
        type=int,
        default=0)
    # parser.add_argument(
    #     '--keep_lowest_n',
    #     type=int,
    #     default=0,
    #     help=
    #     'number of sorted examples to keep that have the lowest score, equivalent to start index of removal, if a negative number given, remove random draw of examples'
    # )
    # parser.add_argument(
    #     '--remove_subsample',
    #     type=int,
    #     default=0,
    #     help='number of examples to remove from the keep-lowest-n examples')
    parser.add_argument(
        '--noise_percent_labels',
        type=int,
        default=0,
        help='percent of labels to randomly flip to a different label')
    parser.add_argument(
        '--noise_percent_pixels',
        type=int,
        default=0,
        help='percent of pixels to randomly introduce Gaussian noise to')
    parser.add_argument(
        '--noise_std_pixels',
        type=float,
        default=0,
        help='standard deviation of Gaussian pixel noise')
    parser.add_argument(
        '--optimizer',
        default="sgd",
        help='optimizer to use, default is sgd. Can also use adam')
    parser.add_argument(
        '--input_dir',
        default='cifar10_results/',
        help='directory where to read sorting file from')
    parser.add_argument(
        '--output_dir', required=True, help='directory where to save results')

    # Parse arguments and setup name of output file with forgetting stats
    args = parser.parse_args()
    args_dict = vars(args)
    print(args_dict)
    main(args)
