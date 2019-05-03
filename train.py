"""
    @Author: Jiang Xin
    @Date: 2019.5.1
"""


from __future__ import print_function, division


import shutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
import time
import os
import copy
from tqdm import tqdm


from model.AlexNet import alexnet


CLASSES = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="PyTorch AlexNet Training")
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--e', dest='epochs', default=30, type=int, help='epochs' )
parser.add_argument('--pretrain', default=True, type=bool, help='if uses the pretrain model to train')
parser.add_argument('--mdir', type=str, dest='model_dir', help='pre-tunning model path')
parser.add_argument('--sdir', type=str, default='checkpoints', dest='save_dir', help='the path to save your model')
parser.add_argument('--bs', type=int, default=8, dest='batch_size', help='train batch size')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)', dest='weight_decay')


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def load(model, model_dir):
    model.load_state_dict(torch.load(model_dir))


def save(model, loss, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file_name = "alexnet_" + str(loss.data) + ".pth"
    torch.save(model.state_dict(), os.path.join(save_dir, file_name))



def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    for _, (x, target) in tqdm(enumerate(train_loader)):
        x = x.cuda(device, non_blocking=True)
        target = target.cuda(device, non_blocking=True)
        output = model(x)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #must add this code,else the training will be slower and slower
        torch.cuda.empty_cache()
    return running_loss

def validate(test_loader, model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for _, (x, target) in tqdm(enumerate(test_loader)):
            x = x.cuda(device, non_blocking=True)
            target = target.cuda(device, non_blocking=True)
            output = model(x)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            torch.cuda.empty_cache()
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def main():
    args= parser.parse_args()
    if args.model_dir:
        if os.path.isfile(args.model_dir):
            alexnet_model = alexnet(num_classes=CLASSES)
            load(alexnet_model, args.model_dir)
        else:
            print('The directory of model is error! Please check out your inputs and try again!')
            return
    else:
        if args.pretrain:
            # because the torchvision pretrain model is based on imgeSet, which has 1000 classes, so we need rebuild the fc
            alexnet_model = alexnet(pretrained=True)
            set_parameter_requires_grad(alexnet_model, True)
            
            alexnet_model.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, CLASSES))
        else:
            alexnet_model = alexnet(num_classes=CLASSES)

    # Data loading code
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=data_transforms['train'])
    test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=data_transforms['val'])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    alexnet_model = alexnet_model.cuda(device)
    criterion = nn.CrossEntropyLoss().cuda(device)
    optimizer = torch.optim.SGD(alexnet_model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch, args)
        running_loss = train(train_loader, alexnet_model, criterion, optimizer, epoch)
        print("epoch:{} use {}s".format(epoch, time.time()-start_time))
        print("The epoch:{} loss is {}".format(epoch, running_loss))
        if epoch % 10 == 0 and epoch > 0:
            validate(test_loader, alexnet_model)
            save(alexnet_model, running_loss, args.save_dir)
        

    validate(test_loader, alexnet_model)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()






