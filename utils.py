import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from functools import reduce
import timeit

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

def reverse_transform(inp, normalization=False):
    inp = inp.numpy().transpose((1, 2, 0))
    if normalization:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp

def train(train_loader, val_loader, device, net, optimizer, scheduler, args):
    for epoch in range(args.epochs):
        start_time = timeit.default_timer()

        # Each epoch has a training and validation
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

    return net

def test(test_loader, device, net, args):
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

def plot_img_array(img_array, ncol=3):
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))

    for i in range(len(img_array)):
        plots[i // ncol, i % ncol]
        plots[i // ncol, i % ncol].imshow(img_array[i])

def masks_to_colorimg(masks):
    colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56)])

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:, y, x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)

def test_sim(test_loader, device, net, args):
    net.eval() # Set model to evaluate mode
    # Get a batch of training data
    for inputs, masks in test_dataloader:
        for name, x  in [('input', inputs.numpy()), ('masks', masks.numpy())]:
            print('%s %s' % (name, x.shape), end='')
            print(' min %5.3f max %5.3f mean %5.3f std %5.3f' % (x.min(), x.max(), x.mean(), x.std()))

        # Left: input image
        plt.subplot(1, 2, 1)
        # Change channel-order and make 3 channels
        plt.imshow(reverse_transform(inputs[2]))

        # Right: targer mask (ground-truth)
        plt.subplot(1, 2, 2)
        # Map each channel (i.e. class) to each color
        plt.imshow(masks_to_colorimg(masks[2]))
        plt.tight_layout()
        plt.savefig('figure/input_sim.png')

        inputs = inputs.to(device)
        masks = masks.to(device)

        # freeze backbone layers
        with torch.no_grad():
            pred = net(inputs)

        pred = pred.data.cpu().numpy()

        # Change channel-order and make 3 channels for matplot
        input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

        # Map each channel (i.e. class) to each color
        target_masks_rgb = [masks_to_colorimg(x) for x in masks.cpu().numpy()]
        pred_rgb = [masks_to_colorimg(x) for x in pred]

        filename = 'figure/output_sim.png'
        img_arrays = [input_images_rgb, target_masks_rgb, pred_rgb]
        flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))

        plot_img_array(np.array(flatten_list), ncol=len(img_arrays))
        plt.tight_layout()
        plt.savefig(filename)

        break
