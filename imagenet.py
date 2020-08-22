'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import numpy as np
from PIL import ImageFile



ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import math
import os
import shutil
import time
import random
from functools import partial

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import _LRScheduler

from attacker import NoOpAttacker, PGDAttacker
import net
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.fastaug.fastaug import FastAugmentation
from utils.fastaug.augmentations import Lighting


def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status


to_clean_status = partial(to_status, status='clean')
to_adv_status = partial(to_status, status='adv')
to_mix_status = partial(to_status, status='mix')

# Models
default_model_names = sorted(name for name in net.__dict__ if name.islower() and not name.startswith('__') and callable(net.__dict__[name]) and not name.startswith("to_") and not name.startswith("partial"))

model_names = default_model_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

# commented by HYC
# the learning rate of the setting 'step' cannot be handled automatically,
# so you should change --lr as you wanted,
# but you don't need to change other settings.
# more information can be referred in the function adjust_learning_rate
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--load', default='', type=str)
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

#Add by YW
parser.add_argument('--warm', default=5, type=int, help='warm up epochs')
parser.add_argument('--warm_lr', default=0.1, type=float, help='warm up start lr')
parser.add_argument('--num_classes', default=200, type=int, help='number of classes')
parser.add_argument('--mixbn', action='store_true', help='use mixbn')
parser.add_argument('--lr_schedule', type=str, default='step', choices=['step', 'cos'])
parser.add_argument('--fastaug', action='store_true')
parser.add_argument('--already224', action='store_true')
# added by HYC, training options, you'd better set smoothing to improve the accuracy.
# but nesterov and lighting make the training too slow and don't have much improvement.
parser.add_argument('--nesterov', action='store_true')
parser.add_argument('--lighting', action='store_true')
parser.add_argument('--smoothing', type=float, default=0)
# added by HYC, attacker options
parser.add_argument('--attack-iter', help='Adversarial attack iteration', type=int, default=0)
parser.add_argument('--attack-epsilon', help='Adversarial attack maximal perturbation', type=float, default=1.0)
parser.add_argument('--attack-step-size', help='Adversarial attack step size', type=float, default=1.0)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc, state
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if args.attack_iter == 0:
        attacker = NoOpAttacker()
    else:
        attacker = PGDAttacker(args.attack_iter, args.attack_epsilon, args.attack_step_size, prob_start_from_clean=0.2 if not args.evaluate else 0.0)

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    
    # the mean and variant don't have too much influence
    # (pic - 0.5) / 0.5 just make it easier for attacker.

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    transform_train = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    if args.fastaug:
        transform_train.transforms.insert(0, FastAugmentation())
    if args.lighting:
        __imagenet_pca = {
            'eigval': np.array([0.2175, 0.0188, 0.0045]),
            'eigvec': np.array([
                [-0.5675,  0.7192,  0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948,  0.4203],
            ])
        }
        transform_train = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
            normalize
        ])
    train_dataset = datasets.ImageFolder(traindir, transform_train)
    train_loader = torch.utils.data.DataLoader((train_dataset),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_transforms = [
            transforms.ToTensor(),
            normalize,
        ]
    if not args.already224:
        val_transforms = [transforms.Scale(256), transforms.CenterCrop(224)] + val_transforms
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose(val_transforms)),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.mixbn:
        norm_layer = MixBatchNorm2d
    else:
        norm_layer = None
    model = net.__dict__[args.arch](num_classes=args.num_classes, norm_layer=norm_layer)
    model.set_attacker(attacker)
    model.set_mixbn(args.mixbn)

    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    if args.smoothing == 0:
        criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    # implement a cross_entropy with label smoothing. 
    # First, perform a log_softmax; then fill the selected classes with 1-smoothing
    # At last, use kl_div, which means:
    # KL(p||q) = -\int p(x)ln q(x) dx - (-\int p(x)ln p(x) dx)
    # kl_div is different from Crossentropy with a const number (\int p(x) ln p(x))
    else:
        criterion = partial(label_smoothing_cross_entropy, classes=args.num_classes, dim=-1)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    # Resume
    title = 'ImageNet-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        if args.load:
            checkpoint = torch.load(args.load)
            if args.mixbn:
                to_merge = {}
                for key in checkpoint['state_dict']:
                    if 'bn' in key:
                        tmp = key.split("bn")
                        aux_key = tmp[0] + 'bn' + tmp[1][0] + '.aux_bn' + tmp[1][1:]
                        to_merge[aux_key] = checkpoint['state_dict'][key]
                    elif 'downsample.1' in key:
                        tmp = key.split("downsample.1")
                        aux_key = tmp[0] + 'downsample.1.aux_bn' + tmp[1]
                        to_merge[aux_key] = checkpoint['state_dict'][key]
                checkpoint['state_dict'].update(to_merge)

            model.load_state_dict(checkpoint['state_dict'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return
    

    # Train and val
    writer = SummaryWriter(log_dir=args.checkpoint)
    warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * args.warm, start_lr=args.warm_lr) if args.warm > 0 else None
    for epoch in range(start_epoch, args.epochs):
        if epoch >= args.warm and args.lr_schedule == 'step':
            adjust_learning_rate(optimizer, epoch, args)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[-1]['lr']))

        train_func = partial(train, train_loader=train_loader, model=model, criterion=criterion,
                             optimizer=optimizer, epoch=epoch, use_cuda=use_cuda,
                             warmup_scheduler=warmup_scheduler, mixbn=args.mixbn,
                             writer=writer, attacker=attacker)
        if args.mixbn:
            model.apply(to_mix_status)
            train_loss, train_acc, loss_main, loss_aux, top1_main, top1_aux = train_func()
        else:
            train_loss, train_acc = train_func()
        writer.add_scalar('Train/loss', train_loss, epoch)
        writer.add_scalar('Train/acc', train_acc, epoch)

        if args.mixbn:
            writer.add_scalar('Train/loss_main', loss_main, epoch)
            writer.add_scalar('Train/loss_aux', loss_aux, epoch)
            writer.add_scalar('Train/acc_main', top1_main, epoch)
            writer.add_scalar('Train/acc_aux', top1_aux, epoch)
            model.apply(to_clean_status)
        test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)
        writer.add_scalar('Test/loss', test_loss, epoch)
        writer.add_scalar('Test/acc', test_acc, epoch)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    print('Best acc:')
    print(best_acc)
    writer.close()
    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))


def train(train_loader, model, criterion, optimizer, epoch, use_cuda, warmup_scheduler, mixbn=False,
          writer=None, attacker=NoOpAttacker()):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    if mixbn:
        losses_main = AverageMeter()
        losses_aux = AverageMeter()
        top1_main = AverageMeter()
        top1_aux = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if epoch < args.warm:
            warmup_scheduler.step()
        elif args.lr_schedule == 'cos':
            adjust_learning_rate(optimizer, epoch, args, batch=batch_idx, nBatch=len(train_loader))

        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        
        # you'd better see the code in net.py to understand what it does when attacker is PGD attacker.
        # the advprop part is done inside forward function.
        # if the advprop part is set outside the forward function, the way to concatenate the batches costs 
        # more time. (around 10 minutes per epoch)
        outputs, targets = model(inputs, targets)
        if args.mixbn:
            outputs = outputs.transpose(1, 0).contiguous().view(-1, args.num_classes)
            targets = targets.transpose(1, 0).contiguous().view(-1)
        loss = criterion(outputs, targets).mean()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), outputs.size(0))
        top1.update(prec1.item(), outputs.size(0))
        top5.update(prec5.item(), outputs.size(0))

        if mixbn:
            with torch.no_grad():
                batch_size = outputs.size(0)
                loss_main = criterion(outputs[:batch_size // 2], targets[:batch_size // 2]).mean()
                loss_aux  = criterion(outputs[batch_size // 2:], targets[batch_size // 2:]).mean()
                prec1_main = accuracy(outputs.data[:batch_size // 2],
                                      targets.data[:batch_size // 2], topk=(1,))[0]
                prec1_aux  = accuracy(outputs.data[batch_size // 2:],
                                      targets.data[batch_size // 2:], topk=(1,))[0]
            losses_main.update(loss_main.item(), batch_size // 2)
            losses_aux.update(loss_aux.item(), batch_size // 2)
            top1_main.update(prec1_main.item(), batch_size // 2)
            top1_aux.update(prec1_aux.item(), batch_size // 2)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if not mixbn:
            loss_str = "{:.4f}".format(losses.avg)
            top1_str = "{:.4f}".format(top1.avg)
        else:
            loss_str = "{:.2f}/{:.2f}/{:.2f}".format(losses.avg, losses_main.avg, losses_aux.avg)
            top1_str = "{:.2f}/{:.2f}/{:.2f}".format(top1.avg, top1_main.avg, top1_aux.avg)
        bar.suffix  = '({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.2f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:s} | top1: {top1:s} | top5: {top5: .1f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=loss_str,
                    top1=top1_str,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    if mixbn:
        return losses.avg, top1.avg, losses_main.avg, losses_aux.avg, top1_main.avg, top1_aux.avg
    else:
        return (losses.avg, top1.avg)

def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        with torch.no_grad():
            outputs, targets = model(inputs, targets)
            loss = criterion(outputs, targets).mean()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, args, batch=None, nBatch=None):
    global state
    if args.lr_schedule == 'cos':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        state['lr'] = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif args.lr_schedule == 'step':
        if epoch in args.schedule:
            state['lr'] *= args.gamma
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']


def label_smoothing_cross_entropy(pred, target, classes, dim, reduction='batchmean', smoothing=0.1):
    '''
    adopted from https://github.com/OpenNMT/OpenNMT-py/blob/e8622eb5c6117269bb3accd8eb6f66282b5e67d9/onmt/utils/loss.py#L186
    and https://github.com/pytorch/pytorch/issues/7455
    '''
    confidence = 1.0-smoothing
    pred = pred.log_softmax(dim=dim)
    with torch.no_grad():
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(smoothing / (classes -1))
        true_dist.scatter_(1, target.data.unsqueeze(1), confidence)
    return F.kl_div(pred, true_dist, reduction=reduction)


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1, start_lr=0.1):
        self.total_iters = total_iters
        self.start_lr = start_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        ret = [self.start_lr + (base_lr - self.start_lr) * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
        return ret


class MixBatchNorm2d(nn.BatchNorm2d):
    '''
    if the dimensions of the tensors from dataloader is [N, 3, 224, 224]
    that of the inputs of the MixBatchNorm2d should be [2*N, 3, 224, 224].

    If you set batch_type as 'mix', this network will using one batchnorm (main bn) to calculate the features corresponding to[:N, 3, 224, 224],
    while using another batch normalization (auxiliary bn) for the features of [N:, 3, 224, 224].
    During training, the batch_type should be set as 'mix'.

    During validation, we only need the results of the features using some specific batchnormalization.
    if you set batch_type as 'clean', the features are calculated using main bn; if you set it as 'adv', the features are calculated using auxiliary bn.

    Usually, we use to_clean_status, to_adv_status, and to_mix_status to set the batch_type recursively. It should be noticed that the batch_type should be set as 'adv' while attacking.
    '''
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MixBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.aux_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                                     track_running_stats=track_running_stats)
        self.batch_type = 'clean'

    def forward(self, input):
        if self.batch_type == 'adv':
            input = self.aux_bn(input)
        elif self.batch_type == 'clean':
            input = super(MixBatchNorm2d, self).forward(input)
        else:
            assert self.batch_type == 'mix'
            batch_size = input.shape[0]
            # input0 = self.aux_bn(input[: batch_size // 2])
            # input1 = super(MixBatchNorm2d, self).forward(input[batch_size // 2:])
            input0 = super(MixBatchNorm2d, self).forward(input[:batch_size // 2])
            input1 = self.aux_bn(input[batch_size // 2:])
            input = torch.cat((input0, input1), 0)
        return input




if __name__ == '__main__':
    main()
