import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import os

from datasets import KidneyDataset
from models import UNet

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device.')

    transform = transforms.Compose([
        transforms.Resize(192),
        transforms.ToTensor(),
    ])

    test_set = KidneyDataset('test', transform=transform)
    test_loader = DataLoader(test_set, batch_size=5, shuffle=False, num_workers=0)

    net = UNet(args.num_classes).to(device)

    if args.model_path and os.path.exists(args.model_path):
        # Load model weights.
        net.load_state_dict(torch.load(args.model_path, map_location=device))

    net.eval()
    for index, (images, masks) in enumerate(test_loader, 1):
        images = images.to(device)

        with torch.no_grad():
            outputs = net(images)

        plt.figure(figsize=(2.5, 2.5))
        plt.imshow(images[4].cpu().numpy().transpose(1, 2, 0))
        plt.tight_layout()
        plt.savefig(f'{args.figure_path}/source.png', dpi=100)

        plt.figure(figsize=(2.5, 2.5))
        plt.imshow(masks[4].numpy().transpose(1, 2, 0))
        plt.tight_layout()
        plt.savefig(f'{args.figure_path}/mask.png', dpi=100)

        plt.figure(figsize=(2.5, 2.5))
        plt.imshow(outputs[4].detach().cpu().numpy().transpose(1, 2, 0))
        plt.tight_layout()
        plt.savefig(f'{args.figure_path}/predicted.png', dpi=100)

        if index == 3:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--figure_path', default='figure', type=str)
    args = parser.parse_args()
    print(vars(args))
    
    os.makedirs(args.figure_path, exist_ok=True)

    main(args)
