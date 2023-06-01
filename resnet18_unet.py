import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import timeit
import os

from datasets import KidneyDataset
from models import ResNetUNet
from utils import train, test

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % (device))

    # use same transform for train/val for this example
    #transform = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    #])
    transform = transforms.Compose([
        #transforms.Resize(192),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # imagenet
    ])

    train_set = KidneyDataset('train', transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    test_set = KidneyDataset('test', transform=transform)
    test_loader = DataLoader(test_set, batch_size=5, shuffle=False, num_workers=0)

    net = ResNetUNet(args.num_classes).to(device)

    if args.model_path and os.path.exists(args.model_path):
        # load model weights
        net.load_state_dict(torch.load(args.model_path, map_location=device))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    net = train(train_loader, None, device, net, optimizer, scheduler, args)
    test(test_loader, device, net, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--step_size', default=25, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    args = parser.parse_args()
    print(vars(args))

    main(args)
