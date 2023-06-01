import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import os

import datasets
import models
from utils import train, test_sim

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % (device))

    # use same transform for train/val for this example
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = datasets.SimDataset(2000, transform=transform)
    val_set = datasets.SimDataset(200, transform=transform)
    test_set = datasets.SimDataset(3, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader =  DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=3, shuffle=False, num_workers=0)

    print('train_set', len(train_set), 'val_set', len(val_set), 'test_set', len(test_set))

    net = models.FCN(args.num_classes).to(device)

    if args.model_path and os.path.exists(args.model_path):
        # load model weights
        net.load_state_dict(torch.load(args.model_path, map_location=device))

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every step_size epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    net = train(train_loader, val_loader, device, net, optimizer, scheduler, args)
    test_sim(test_loader, device, net, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--num_classes', default=6, type=int)
    parser.add_argument('--step_size', default=7, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    args = parser.parse_args()
    print(vars(args))

    main(args)
