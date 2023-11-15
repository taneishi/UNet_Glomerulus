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

def dice_loss(y_pred, y_true, smooth=1e-5):
    y_pred = y_pred.contiguous()
    y_true = y_true.contiguous()

    intersection = (y_pred * y_true).sum(dim=2).sum(dim=2)
    union = y_pred.sum(dim=2).sum(dim=2) + y_true.sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (union + smooth)))

    return loss.mean()

def criterion(y_pred, y_true, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(y_pred, y_true)

    y_pred = torch.sigmoid(y_pred)
    dice = dice_loss(y_pred, y_true)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] = metrics.get('bce', 0) + bce.data.cpu().numpy() * y_true.size(0)
    metrics['dice'] = metrics.get('dice', 0) + dice.data.cpu().numpy() * y_true.size(0)
    metrics['loss'] = metrics.get('loss', 0) + loss.data.cpu().numpy() * y_true.size(0)

    return loss

def plot(test_loader, device, net):
    os.makedirs('figure', exist_ok=True)

    net.eval()
    for index, (images, masks) in enumerate(test_loader, 1):
        images = images.to(device)
        with torch.no_grad():
            outputs = net(images)

        print(outputs.shape)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(images[4].cpu().numpy().transpose(1, 2, 0))

        plt.subplot(1, 3, 2)
        plt.imshow(masks[4].numpy().transpose(1, 2, 0))

        plt.subplot(1, 3, 3)
        plt.imshow(outputs[4].detach().cpu().numpy().transpose(1, 2, 0))

        plt.tight_layout()
        plt.savefig('figure/output%02d.png' % (index))

        if index == 3:
            break

    return net

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % (device))

    transform = transforms.Compose([
        transforms.Resize(192),
        transforms.ToTensor(),
    ])

    train_set = KidneyDataset('train', transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_loader = None

    test_set = KidneyDataset('test', transform=transform)
    test_loader = DataLoader(test_set, batch_size=5, shuffle=False, num_workers=0)

    net = UNet(args.num_classes).to(device)

    if args.model_path and os.path.exists(args.model_path):
        # Load model weights.
        net.load_state_dict(torch.load(args.model_path, map_location=device))

    # Observe all parameters to be optimized.
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    for epoch in range(args.epochs):
        start_time = timeit.default_timer()

        # Each epoch has a training.
        metrics = dict()
        net.train() # Set model to training mode.
        for index, (images, masks) in enumerate(train_loader, 1):
            images = images.to(device)
            masks = masks.to(device)

            # forward
            outputs = net(images)
            loss = criterion(outputs, masks, metrics)

            optimizer.zero_grad() # Set zero to the parameter gradients.
            loss.backward()
            optimizer.step()

        print('\repoch %3d/%3d batch %3d/%3d train ' % (epoch+1, args.epochs, index, len(train_loader)), end='')
        print(' '.join(['%s %5.3f' % (k, metrics[k] / (index * args.batch_size)) for k in metrics.keys()]), end='')
        
        scheduler.step()
        for param_group in optimizer.param_groups:
            print(' lr %1.0e' % (param_group['lr']), end='')

        # Perform a validation if validation data was set.
        if val_loader:
            metrics = dict()
            net.eval() # Set model to evaluate mode
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                with torch.no_grad():
                    outputs = net(images)

            loss = criterion(outputs, masks, metrics)

            print(' val ', end='')
            print(' '.join(['%s %5.3f' % (k, metrics[k] / len(val_loader.dataset)) for k in metrics.keys()]), end='')

        print(' %4.1fsec' % (timeit.default_timer() - start_time))

        if args.model_path:
            torch.save(net.state_dict(), args.model_path)

    plot(test_loader, device, net)

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
    
    if args.model_path:
        os.makedirs(args.model_path, exist_ok=True)

    main(args)
