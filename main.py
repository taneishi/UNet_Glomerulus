import numpy as np
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

def dice_loss(y_pred, y_true, smooth=1e-5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.contiguous()
    y_true = y_true.contiguous()

    intersection = (y_pred * y_true).sum(dim=2).sum(dim=2)
    union = y_pred.sum(dim=2).sum(dim=2) + y_true.sum(dim=2).sum(dim=2)

    dice = (2. * intersection + smooth) / (union + smooth)
    loss = 1 - dice

    return loss.mean()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device.')

    transform = transforms.Compose([
        transforms.Resize(192),
        transforms.ToTensor(),
    ])

    train_set = KidneyDataset('train', transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    net = UNet(args.num_classes).to(device)

    if args.model_path and os.path.exists(args.model_path):
        # Load model weights.
        net.load_state_dict(torch.load(args.model_path, map_location=device))

    # Observe all parameters to be optimized.
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    for epoch in range(args.epochs):
        start_time = timeit.default_timer()

        train_bce, train_dice, train_loss = 0, 0, 0
        # Each epoch has a training. Set model to training mode.
        net.train()
        for index, (images, masks) in enumerate(train_loader, 1):
            images = images.to(device)
            masks = masks.to(device)

            outputs = net(images)

            bce = F.binary_cross_entropy_with_logits(outputs, masks)
            dice = dice_loss(outputs, masks)
            loss = bce * args.bce_weight + dice * (1 - args.bce_weight)

            train_bce += bce.item()
            train_dice += dice.item()
            train_loss += loss.item()

            # Set zero to the parameter gradients.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'epoch {epoch+1:3d}/{args.epochs:3d} batch {index:3d}/{len(train_loader):3d} train', end='')
        print(f' bce {train_bce / len(train_loader):5.3f}', end='')
        print(f' dice {train_dice / len(train_loader):5.3f}', end='')
        print(f' loss {train_loss / len(train_loader):5.3f}', end='')
        
        scheduler.step()
        for param_group in optimizer.param_groups:
            print(f' lr {param_group["lr"]:1.0e}', end='')

        print(f' {timeit.default_timer() - start_time:4.1f}sec')

        if args.model_path:
            torch.save(net.state_dict(), args.model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--step_size', default=25, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--bce_weight', default=0.5, type=float)
    args = parser.parse_args()
    print(vars(args))
    
    if args.model_path:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    main(args)
