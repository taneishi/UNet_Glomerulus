import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import timeit
import os

from datasets import KidneyDataset
from models import UNet
from utils import train, test

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % (device))

    transform = transforms.Compose([
        transforms.Resize(192),
        transforms.ToTensor(),
    ])

    train_set = KidneyDataset('train', transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    test_set = KidneyDataset('test', transform=transform)
    test_loader = DataLoader(test_set, batch_size=5, shuffle=False, num_workers=0)

    net = UNet(args.num_classes).to(device)

    if args.model_path and os.path.exists(args.model_path):
        # load model weights
        net.load_state_dict(torch.load(args.model_path, map_location=device))

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
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
    parser.add_argument('--lr', default=5e-5, type=float)
    args = parser.parse_args()
    print(vars(args))

    main(args)
