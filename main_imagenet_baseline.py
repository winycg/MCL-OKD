import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np


import models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds


from bisect import bisect_right
import time
import math

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='./data/', type=str, help='trainset directory')
parser.add_argument('--arch', default='resnet34', type=str, help='network architecture')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr-type', default='SGDR', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[30, 60, 90], type=list, help='milestones for lr-multistep')
parser.add_argument('--sgdr-t', default=100, type=int, dest='sgdr_t',help='SGDR T_0')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=256, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')

# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# -----------------------------------------------------------------------------------------
# dataset
num_classes = 1000
train_set = datasets.ImageFolder(
    os.path.join(args.data, 'train'),
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]))

test_set = datasets.ImageFolder(
    os.path.join(args.data, 'val'), transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
]))

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True)
# --------------------------------------------------------------------------------------------

# Model
print('==> Building model..')
model = getattr(models, args.arch)
net = model(num_classes=num_classes)
print('Params: %.2fM, Multi-adds: %.2fG'
      % (cal_param_size(net)/1e6, cal_multi_adds(net, (2, 3, 224, 224))/1e9))
del(net)

net = model(num_classes=num_classes).cuda()
net = torch.nn.DataParallel(net)
cudnn.benchmark = True


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum()
        res.append(correct_k)
    return res


def adjust_lr(optimizer, epoch, eta_max=0.1, eta_min=0.):
    cur_lr = 0.
    if args.lr_type == 'SGDR':
        i = int(math.log2(epoch / args.sgdr_t + 1))
        T_cur = epoch - args.sgdr_t * (2 ** (i) - 1)
        T_i = (args.sgdr_t * 2 ** i)

        cur_lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * T_cur / T_i))

    elif args.lr_type == 'multistep':
        cur_lr = args.init_lr * 0.1 ** bisect_right(args.milestones, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr


# Training
def train(epoch, criterion_list, optimizer):
    train_loss = 0.

    acc1 = 0.
    acc5 = 0.
    total = 0

    lr = adjust_lr(optimizer, epoch)
    start_time = time.time()
    criterion_cls = criterion_list[0]

    net.train()
    batch_start_time = time.time()
    for batch_idx, (input, target) in enumerate(train_loader):
        input = input.float()
        input = input.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        logit = net(input)

        loss = criterion_cls(logit, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(train_loader)

        _, predicted = logit.max(1)
        prec1, prec5 = accuracy(logit, target, topk=(1, 5))
        total += target.size(0)
        acc1 += prec1
        acc5 += prec5

        print('Batch:{}, Time:{:.3f}, acc1:{:.2f}, acc5:{:.2f}'
              .format(batch_idx,
                      time.time() - batch_start_time,
                      prec1.item()/target.size(0),
                      prec1.item()/target.size(0)))

        batch_start_time = time.time()

    acc1 = (acc1 / total * 100).item()
    acc5 = (acc5 / total * 100).item()
    with open('result/' + str(os.path.basename(__file__).split('.')[0]) + args.arch + '.txt', 'a+') as f:
        f.write('Epoch:{}\t lr:{:.3f}\t duration:{:.3f}'
                '\n train_loss:{:.5f}'
                '\t acc1: {:.2f} \t acc5: {:.2f} '
                .format(epoch, lr, time.time() - start_time,
                        train_loss, acc1, acc5))


def test(epoch, criterion_cls):
    net.eval()
    global best_acc
    test_loss_cls = 0.

    acc1 = 0.
    acc5 = 0.
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(test_loader):
            inputs, target = inputs.cuda(), target.cuda()
            logit= net(inputs)

            loss_cls = 0.
            loss_cls = loss_cls + criterion_cls(logit, target)

            test_loss_cls += loss_cls.item()/ len(test_loader)
            _, predicted = logit.max(1)
            prec1, prec5 = accuracy(logit, target, topk=(1, 5))
            total += target.size(0)
            acc1 += prec1
            acc5 += prec5

        acc1 = acc1 / total * 100
        acc5 = acc5 / total * 100

        with open('result/' + str(os.path.basename(__file__).split('.')[0]) + args.arch + '.txt', 'a+') as f:
            f.write('test epoch:{}\t test_loss_cls:{:.5f}'
                    '\t acc1:{:.2f}\t acc5:{:.2f}\n'
                    .format(epoch, test_loss_cls, acc1.item(), acc5.item()))
        print('test epoch:{}\t acc1:{:.2f}\t acc5:{:.2f}\n'
              .format(epoch, acc1.item(), acc5.item()))

    return acc1


def main():
    global best_acc
    global start_epoch
    criterion_cls = nn.CrossEntropyLoss()

    if args.evaluate:
        checkpoint = torch.load('./checkpoint/' + model.__name__ + '_best.pth.tar',
                                map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        test(start_epoch, criterion_cls)
    else:
        trainable_list = nn.ModuleList([])
        trainable_list.append(net)

        optimizer = optim.SGD(trainable_list.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)

        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_cls)  # classification loss
        criterion_list.cuda()

        if args.resume:
            checkpoint = torch.load('./checkpoint/' + model.__name__ + '.pth.tar', map_location=torch.device('cpu'))
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']

        for epoch in range(start_epoch, args.epochs):
            train(epoch, criterion_list, optimizer)
            acc = test(epoch, criterion_cls)

            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/' + str(model.__name__) + '.pth.tar')

            is_best = False
            if best_acc < acc:
                best_acc = acc
                is_best = True

            if is_best:
                shutil.copyfile('./checkpoint/' + str(model.__name__) + '.pth.tar',
                                './checkpoint/' + str(model.__name__) + '_best.pth.tar')


if __name__ == '__main__':
    main()
