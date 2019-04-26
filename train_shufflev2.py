'''Train CK+ with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms,utils
from torch.utils.data import DataLoader
import transforms as transforms
import numpy as np
import os
import argparse
import utils
from CK import CK
from torch.autograd import Variable
from models import *
from ShuffleNetV2 import ShuffleNetV2


train_data = './test_img/train'
test_data = './test_img/eval'

parser = argparse.ArgumentParser(description='PyTorch CK+ CNN Training')
#parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='CK+', help='dataset')
parser.add_argument('--fold', default=1, type=int, help='k fold number')
parser.add_argument('--bs', default=128, type=int, help='batch_size')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_Test_acc = 0  # best PrivateTest accuracy
best_Test_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 20  # 50
learning_rate_decay_every = 1 # 5
learning_rate_decay_rate = 0.8 # 0.9

cut_size = 44
total_epoch = 200
bs = 64

#path = os.path.join(opt.dataset + '_' + opt.model, str(opt.fold))
path = './CK+ShuffleNetV2/'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(cut_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

trainset=torchvision.datasets.ImageFolder(train_data,transform_train)
trainloader=DataLoader(trainset,bs,shuffle=True)
testset=torchvision.datasets.ImageFolder(test_data,transform_test)
testloader=DataLoader(testset,batch_size=5,shuffle=True)

#trainset = CK(split = 'Training', fold = opt.fold, transform=transform_train)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=1)
#testset = CK(split = 'Testing', fold = opt.fold, transform=transform_test)
#testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False, num_workers=1)

# Model
'''
if opt.model == 'VGG19':
    net = VGG('VGG19')
elif opt.model == 'Resnet18':
    net = ResNet18()
'''
net = ShuffleNetV2(n_class=7)
#net = net.to(device=my_device)


if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path,'ShuffleNetV2.pth'))
    
    net.load_state_dict(checkpoint['net'])
    best_Test_acc = checkpoint['best_Test_acc']
    best_Test_acc_epoch = checkpoint['best_Test_acc_epoch']
    start_epoch = best_Test_acc_epoch + 1
else:
    print('==> Building model..')

if use_cuda:
    net.cuda()

'''
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
'''
criterion = nn.CrossEntropyLoss()
#criterion=nn.MultiMarginLoss()
optimizer=optim.AdaBound(net.parameters(), lr=opt.lr, final_lr=0.1)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    '''
    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))
    '''

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    Train_acc = 100.*correct/total

# test
def test(epoch):
    global Test_acc
    global best_Test_acc
    global best_Test_acc_epoch
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops

        loss = criterion(outputs_avg, targets)
        PrivateTest_loss += loss.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # Save checkpoint.
    Test_acc = 100.*correct/total

    if Test_acc > best_Test_acc:
        print('Saving..')
        print("best_Test_acc: %0.3f" % Test_acc)
        state = {'net': net.state_dict() if use_cuda else net,
            'best_Test_acc': Test_acc,
            'best_Test_acc_epoch': epoch,
        }
        if not os.path.isdir(opt.dataset + '_' + 'ShuffleNetV2'):
            os.mkdir(opt.dataset + '_' + 'ShuffleNetV2')
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'ShuffleNetV2.pth'))
        best_Test_acc = Test_acc
        best_Test_acc_epoch = epoch

for epoch in range(start_epoch, total_epoch):
    train(epoch)
    test(epoch)

print("best_Test_acc: %0.3f" % best_Test_acc)
print("best_Test_acc_epoch: %d" % best_Test_acc_epoch)