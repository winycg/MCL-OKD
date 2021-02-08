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
import torchvision
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds


from bisect import bisect_right
import time
import math

from mcl.criterion import MCL_Loss
from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--data', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset name')
parser.add_argument('--arch', default='mcl_okd_resnet32_b', type=str, help='network architecture')
parser.add_argument('--init-lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--lr-type', default='SGDR', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[150, 200, 250], type=list, help='milestones for lr-multistep')
parser.add_argument('--sgdr-t', default=300, type=int, dest='sgdr_t',help='SGDR T_0')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='batch size')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')

parser.add_argument('--num-branches', type=int, default=4, help='number of branches')

parser.add_argument('--kd_T', type=float, default=3, help='temperature for KD distillation')
parser.add_argument('--gamma', type=float, default=1., help='weight for classification')
parser.add_argument('-a', '--alpha', type=float, default=1., help='weight balance for KD')
parser.add_argument('-b', '--beta', type=float, default=0.025, help='weight balance for other losses')

parser.add_argument('--rep-dim', default=1024, type=int, help='penultimate dimension')
parser.add_argument('--feat-dim', default=128, type=int, help='feature dimension')
parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
parser.add_argument('--nce_t', default=0.1, type=float, help='temperature parameter for softmax')
parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')


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
num_classes = 100
trainloader, testloader, n_data = get_cifar100_dataloaders_sample(data_folder=args.data,
                                                                   batch_size=args.batch_size,
                                                                   num_workers=args.num_workers,
                                                                   k=args.nce_k,
                                                                   mode=args.mode)
# --------------------------------------------------------------------------------------------

# Model
print('==> Building model..')
model = getattr(models, args.arch)
net = model(num_classes=num_classes, number_net=args.num_branches)
print('Params: %.2fM, Multi-adds: %.2fG'
      % (cal_param_size(net)/1e6, cal_multi_adds(net, (2, 3, 32, 32))/1e9))
del(net)

net = model(num_classes=num_classes, number_net=args.num_branches).cuda()
net = torch.nn.DataParallel(net)
cudnn.benchmark = True


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss


def adjust_lr(optimizer, epoch, eta_max=args.init_lr, eta_min=0.):
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
    train_loss_cls = 0.
    train_loss_div = 0.
    train_loss_kd = 0.

    correct = [0] * args.num_branches
    total = [0] * args.num_branches

    lr = adjust_lr(optimizer, epoch)
    start_time = time.time()
    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    net.train()
    for batch_idx, (input, target, index, contrast_idx) in enumerate(trainloader):
        input = input.float()
        input = input.cuda()
        target = target.cuda()
        index = index.cuda()
        contrast_idx = contrast_idx.cuda()

        optimizer.zero_grad()
        logits, embedding = net(input)

        loss_cls = 0.
        loss_div = 0.
        loss_kd = 0.
        ensemble_logits = 0.
        for i in range(len(logits)):
            loss_cls = loss_cls + criterion_cls(logits[i], target)
        for i in range(len(logits)):
            ensemble_logits = ensemble_logits + logits[i]
        ensemble_logits = ensemble_logits / (len(logits))
        ensemble_logits = ensemble_logits.detach()
        loss_div = loss_div + criterion_div(logits[-1], ensemble_logits)
        loss_kd = loss_kd + criterion_kd(embedding, index, contrast_idx)

        loss = args.gamma * loss_cls + args.alpha * loss_div + args.beta * loss_kd
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(trainloader)
        train_loss_cls += args.gamma * loss_cls.item() / len(trainloader)
        train_loss_div += args.alpha * loss_div.item()/ len(trainloader)
        train_loss_kd += args.beta * loss_kd.item() / len(trainloader)

        for i in range(len(logits)):
            _, predicted = logits[i].max(1)
            correct[i] += predicted.eq(target).sum().item()
            total[i] += target.size(0)

    acc = []
    for i in range(args.num_branches):
        acc.append(correct[i] / total[i])

    with open('result/' + str(os.path.basename(__file__).split('.')[0]) +
              args.arch + '_' + str(args.beta) + '_' +
              str(args.nce_k) + '_' + str(args.num_branches)
              + '_' + str(args.manual_seed) + '.txt', 'a+') as f:
        f.write('Epoch:{0}\t lr:{1:.3f}\t duration:{2:.3f}'
                '\n train_loss:{3:.5f}\t train_loss_cls:{4:.5f}'
                '\t train_loss_div:{5:.5f}\t train_loss_kd:{6:.5f}'
                '\n accuracy: {7} \n'
                .format(epoch, lr, time.time() - start_time,
                        train_loss, train_loss_cls,
                        train_loss_div, train_loss_kd, str(acc)))


def test(epoch, criterion_cls, criterion_div):
    net.eval()
    global best_acc
    test_loss_cls = 0.
    test_loss_div = 0.

    correct = [0] * (args.num_branches + 1)
    total = [0] * (args.num_branches + 1)

    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(testloader):
            inputs, target = inputs.cuda(), target.cuda()
            logits, embedding = net(inputs)

            loss_cls = 0.
            loss_div = 0.
            ensemble_logits = 0.
            for i in range(len(logits)):
                loss_cls = loss_cls + criterion_cls(logits[i], target)
            for i in range(len(logits)):
                ensemble_logits = ensemble_logits + logits[i]
            ensemble_logits = ensemble_logits / (len(logits))
            ensemble_logits = ensemble_logits.detach()
            loss_div = loss_div + criterion_div(logits[-1], ensemble_logits)

            test_loss_cls += loss_cls.item()/ len(testloader)
            test_loss_div += loss_div.item() / len(testloader)

            for i in range(args.num_branches+1):
                if i == args.num_branches:
                    _, predicted = ensemble_logits.max(1)
                else:
                    _, predicted = logits[i].max(1)
                correct[i] += predicted.eq(target).sum().item()
                total[i] += target.size(0)

        acc = []
        for i in range(args.num_branches+1):
            acc.append(correct[i] / total[i])

        with open('result/' + str(os.path.basename(__file__).split('.')[0]) +
                  args.arch + '_' + str(args.beta) + '_' +
                  str(args.nce_k) + '_' + str(args.num_branches)
                  + '_' + str(args.manual_seed) + '.txt', 'a+') as f:
            f.write('test epoch:{0}\t test_loss_cls:{1:.5f}\t test_loss_div:{2:.5f}\t accuracy:{3}\n'
                    .format(epoch, test_loss_cls, test_loss_div, str(acc)))
        print('test epoch:{0}\t accuracy:{1}\n'.format(epoch, str(acc)))

    return max(acc)

    # Save checkpoint

def main():
    global best_acc
    global start_epoch
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)

    if args.evaluate:
        checkpoint = torch.load('./checkpoint/' + model.__name__ + '_best.pth.tar',
                                map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        test(start_epoch, criterion_cls, criterion_div)
    else:
        trainable_list = nn.ModuleList([])
        trainable_list.append(net)

        data = torch.randn(2, 3, 32, 32)
        net.eval()
        logits, embedding = net(data)

        args.rep_dim = embedding[0].shape[1]
        args.n_data = n_data

        criterion_kd = MCL_Loss(args)
        trainable_list.append(criterion_kd.embed_list)
        optimizer = optim.SGD(trainable_list.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_cls)  # classification loss
        criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
        criterion_list.append(criterion_kd)  # other knowledge distillation loss
        criterion_list.cuda()

        if args.resume:
            checkpoint = torch.load('./checkpoint/' + model.__name__ + '.pth.tar', map_location=torch.device('cpu'))
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']

        for epoch in range(start_epoch, args.epochs):
            train(epoch, criterion_list, optimizer)
            acc = test(epoch, criterion_cls, criterion_div)

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
