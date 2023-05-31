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

    metrics['bce'] =  metrics.get('bce', 0) + bce.data.cpu().numpy() * y_true.size(0)
    metrics['dice'] = metrics.get('dice', 0) + dice.data.cpu().numpy() * y_true.size(0)
    metrics['loss'] = metrics.get('loss', 0) + loss.data.cpu().numpy() * y_true.size(0)

    return loss

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % (device))

    transform = transforms.Compose([
        transforms.Resize(192),
        transforms.ToTensor(),
    ])

    train_set = KidneyDataset('train', transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    net = UNet(args.num_classes).to(device)

    if args.model_path and os.path.exists(args.model_path):
        net.load_state_dict(torch.load(args.model_path, map_location=device))

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    for epoch in range(args.epochs):
        start_time = timeit.default_timer()

        metrics = dict()
        net.train() # Set model to training mode
        for index, (images, masks) in enumerate(train_loader, 1):
            images = images.to(device)
            masks = masks.to(device)

            # forward
            outputs = net(images)
            loss = criterion(outputs, masks, metrics)

            optimizer.zero_grad() # zero the parameter gradients
            loss.backward()
            optimizer.step()

            print('\repoch %3d/%3d batch %3d/%3d train ' % (epoch+1, args.epochs, index, len(train_loader)), end='')
            print(' '.join(['%s %5.3f' % (k, metrics[k] / (index * args.batch_size)) for k in metrics.keys()]), end='')

        scheduler.step()
        for param_group in optimizer.param_groups:
            print(' lr %1.0e' % (param_group['lr']), end='')

        print(' %4.1fsec' % (timeit.default_timer() - start_time))

        if args.model_path:
            torch.save(net.state_dict(), args.model_path)

    test_set = KidneyDataset('test', transform=transform)
    test_loader = DataLoader(test_set, batch_size=5, shuffle=False, num_workers=0)

    for images, masks in test_loader:
        images = images.to(device)
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
        plt.savefig('figure/output.png')

        break

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
