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
parser.add_argument('--mdir', type=str, dest='model_dir', help='pre-tunning model path')
parser.add_argument('--bs', type=int, default=8, dest='batch_size', help='train batch size')


def load(model, model_dir):
    model.load_state_dict(torch.load(model_dir))
    print("load model successful")


def save(model, loss, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file_name = "alexnet_" + str(loss.data) + ".pth"
    torch.save(model.state_dict(), os.path.join(save_dir, file_name))


def main():
    args= parser.parse_args()
    if args.model_dir:
        if os.path.isfile(args.model_dir):
            alexnet_model = alexnet(num_classes=CLASSES)
            load(alexnet_model, args.model_dir)
        else:
            print('The directory of model is error! Please check out your inputs and try again!')
            return

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
    test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=data_transforms['val'])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    alexnet_model = alexnet_model.cuda(device)
    alexnet_model.eval()
    total = 0.
    top1 = 0.
    top5 = 0.
    for _, (x, target) in tqdm(enumerate(test_loader)):
        x = x.cuda(device, non_blocking=True)
        target = target.cuda(device, non_blocking=True)
        output = alexnet_model(x)
        total += x.size(0)
        _, predicted = torch.max(output.data, 1)
        top1 += (predicted == target).sum().item()
        _, pred = output.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        top5 += correct[:5].view(-1).sum(0, keepdim=True).cpu()
        torch.cuda.empty_cache()
    top5 = top5.numpy()[0]
    print("the top1 is {} , the top5 is {}".format(top1/total, top5/total))


if __name__ == "__main__":
    main()
