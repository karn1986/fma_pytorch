"""
@author: KAgarwal

Adapted from one of the assignments of 
GaTech's CS 7643 Deep Learning course 
"""
import yaml
import argparse
import time
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import torch
import torchvision
from data import audio_clips

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from models import MyModel

parser = argparse.ArgumentParser(description='FMA')
parser.add_argument('--config', default='configs/config_mymodel.yaml')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]

    _, pred = torch.max(output, dim=-1)

    correct = pred.eq(target).sum() * 1.0

    acc = correct / batch_size

    return acc

def train(epoch, data_loader, model, optimizer, criterion, writer):

    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    for idx, (data, target) in enumerate(data_loader):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
            
        #https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#train-the-network
        # zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        out = model(data)
        # Calculate loss
        loss = criterion(out, target)
        # backward pass to compute gradients
        loss.backward()
        # update model parameters
        optimizer.step()


        batch_acc = accuracy(out, target)

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])
        writer.add_scalar('training loss', loss, epoch * len(data_loader) + idx)
        writer.add_scalar('training accuracy', batch_acc, epoch * len(data_loader) + idx)
        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t')
                   .format(epoch, idx, len(data_loader), iter_time=iter_time, loss=losses, top1=acc))


def validate(epoch, val_loader, model, criterion, writer):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    num_class = 16
    cm =torch.zeros(num_class, num_class)
    # evaluation loop
    for idx, (data, target) in enumerate(val_loader):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        #https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#test-the-network-on-the-test-data
        with torch.no_grad():
            out = model(data)
            loss = criterion(out, target)

        batch_acc = accuracy(out, target)
        writer.add_scalar('Validation loss', loss, epoch * len(val_loader) + idx)
        writer.add_scalar('Validation accuracy', batch_acc, epoch * len(val_loader) + idx)
        # update confusion matrix
        _, preds = torch.max(out, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
               'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t')
               .format(epoch, idx, len(val_loader), iter_time=iter_time, loss=losses, top1=acc))
    cm = cm / cm.sum(1)
    per_cls_acc = cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    print("* Prec @1: {top1.avg:.4f}".format(top1=acc))
    return acc.avg, cm


def adjust_learning_rate(optimizer, epoch, args):
    epoch += 1
    if epoch <= args.warmup:
        lr = args.learning_rate * epoch / args.warmup
    elif epoch > args.steps[1]:
        lr = args.learning_rate * 0.01
    elif epoch > args.steps[0]:
        lr = args.learning_rate * 0.1
    else:
        lr = args.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    # transform_train = transforms.Compose([
    #     transforms.Resize((128,128))
    #     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # Normalize the test set same as training set without augmentation
    # transform_val = transforms.Compose([
    #     transforms.Resize((128,128))
    #     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    train_dataset = audio_clips(audio_dir = args.audio_dir,
                                meta_dir = args.meta_dir,
                                transform = None,
                                mode = 'train')
    enc = train_dataset.enc
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = audio_clips(audio_dir = args.audio_dir,
                              meta_dir = args.meta_dir,
                              transform = None,
                              mode = 'val',
                              enc = enc)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    if args.model == 'MyModel':
        model = MyModel()
    elif args.model == 'ResNet-32':
        # model = resnet32()
        pass
    print(model)
    if torch.cuda.is_available():
        model = model.cuda()

    if args.loss_type == "CE":
        criterion = nn.CrossEntropyLoss()
    else:
        # criterion = FocalLoss(weight=per_cls_weights, gamma=1)
        pass
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.reg)
    best = 0.0
    best_cm = None
    best_model = None
    writer = SummaryWriter('runs/mymodel')
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train loop
        train(epoch, train_loader, model, optimizer, criterion, writer)

        # validation loop
        acc, cm = validate(epoch, val_loader, model, criterion, writer)

        if acc > best:
            best = acc
            best_cm = cm
            best_model = copy.deepcopy(model)

    print('Best Prec @1 Acccuracy: {:.4f}'.format(best))
    per_cls_acc = best_cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    if args.save_best:
        torch.save(best_model.state_dict(), './checkpoints/' + args.model.lower() + '.pth')


























if __name__ == '__main__':
    main()