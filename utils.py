import time
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import torch
from models import *
import numpy as np
import torchvision.utils as vutils
import copy
from torchsummary import summary


def matrix_combinatorial(A, gamma):
    matrix_comb = torch.zeros(A.shape[1], A.shape[1])

    for i in range(A.shape[1]):
        for j in range(A.shape[1]):
            # print(A[i].shape, B[j].shape)
            matrix_comb[i, j] = torch.exp(- gamma * torch.linalg.norm(A[:, i] - A[:, j]))

    return matrix_comb


def get_dataset(dataset, data_path='./data', batch_size=128):
    if dataset == 'MNIST':
        channel = 1
        im_size = (32, 32)
        num_classes = 10
        transform = transforms.Compose([transforms.Resize(im_size), transforms.ToTensor()])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)  # no augmentation
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = (32, 32)
        num_classes = 10
        transform = transforms.Compose([transforms.Resize(im_size), transforms.ToTensor()])
        dst_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)  # no augmentation
        dst_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    else:
        exit('unknown dataset: %s' % dataset)

    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=batch_size, shuffle=False, num_workers=0)
    return channel, im_size, num_classes, class_names, dst_train, dst_test, testloader, trainloader


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def save_metrics(save_path):
    tb_path = save_path + '/tensorboard'
    images_path = save_path + '/images'
    model_path = save_path + '/model'
    metrics_path = save_path + '/metrics'

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(tb_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)
    writer = SummaryWriter(tb_path)
    return writer, images_path, model_path, metrics_path


def one_epoch(mode, dataloader, net, optimizer, criterion, epoch, epochs, SSIM, PSNR,
              writer=None, optical_layer=None, args=None):
    color = 'red'
    t_loss = AverageMeter()
    t_psnr = AverageMeter()
    t_ssim = AverageMeter()
    net = net.to(args.device)

    if mode == 'train':
        net.train()
    else:
        color = 'green'
        net.eval()

    data_loop = tqdm(enumerate(dataloader), total=len(dataloader), colour=color)

    for _, data in data_loop:
        inputs, _ = data
        inputs = inputs.to(args.device)

        if optical_layer is not None:
            pred = net(optical_layer(inputs))
        else:
            pred = net(inputs)

        loss = criterion(pred, inputs)

        batch_psnr = PSNR(inputs, pred)
        batch_ssim = SSIM(inputs, pred)

        t_loss.update(loss.item(), inputs.size(0))
        t_psnr.update(batch_psnr.item(), inputs.size(0))
        t_ssim.update(batch_ssim.item(), inputs.size(0))

        dict_metrics = dict(loss=t_loss.avg, psnr=t_psnr.avg, ssim=t_ssim.avg)

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        data_loop.set_description(f'{mode} Epoch [{epoch + 1} / {epochs}]')
        data_loop.set_postfix(**dict_metrics)

        if writer is not None:
            for key, value in dict_metrics.items():
                writer.add_scalar(f'{mode}_{key}', value, epoch)

    return t_psnr.avg, t_ssim.avg


def save_npy_metric(file, metric_name):
    with open(f'{metric_name}.npy', 'wb') as f:
        np.save(f, file)


def save_aperture_codes(optical_layer, row, pad, path, name):
    aperture_codes = copy.deepcopy(optical_layer.weights.detach())
    aperture_codes = BinaryQuantize.apply(aperture_codes)
    aperture_codes = aperture_codes.unsqueeze(1)
    grid = vutils.make_grid(aperture_codes, nrow=row, padding=pad, normalize=True)
    vutils.save_image(grid, f'{path}/{name}.png')

# ------------------------------------------------------------------------------------------------------------#

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_base_model_unfolding(k, epochs, save_path, SSIM, PSNR, args):
    channel, im_size, num_classes, class_names, dst_train, dst_test, testloader, trainloader = get_dataset(
        args.dataset, args.data_path, batch_size=args.batch_size)

    psnr_train = np.zeros(epochs)
    psnr_test = np.zeros(epochs)
    ssim_train = np.zeros(epochs)
    ssim_test = np.zeros(epochs)

    e2e = E2E_Unfolding_Base(k, im_size[0], im_size[1], channel, n_stages=args.n_stages, binary=args.binary).to(args.device)
    lr = args.lr_baseline
    # p = count_parameters(e2e)
    # print('Number of parameters: ',p)
    summary(e2e, (1, 32, 32))
    e2e_optim = torch.optim.Adam(e2e.parameters(), lr=lr)
    criterion = nn.MSELoss()

    writer, images_path, model_path, metrics_path = save_metrics(save_path)

    for i in range(0, epochs):
        train_psnr, train_ssim = one_epoch("train", dataloader=trainloader, net=e2e, optimizer=e2e_optim,
                                           criterion=criterion,
                                           epoch=i, epochs=epochs, SSIM=SSIM, PSNR=PSNR, writer=writer, args=args)
        test_psnr, test_ssim = one_epoch("test", dataloader=testloader, net=e2e, optimizer=None, criterion=criterion,
                                         epoch=i, epochs=epochs, SSIM=SSIM, PSNR=PSNR, writer=writer, args=args)

        psnr_train[i] = train_psnr
        psnr_test[i] = test_psnr
        ssim_test[i] = test_ssim
        ssim_train[i] = train_ssim
        torch.save(e2e.state_dict(),
                   f'{model_path}/{i + 1}_lr_{lr}_k_{k}_{np.round(train_psnr, 2)}_{np.round(test_psnr, 2)}.pth')

    save_aperture_codes(e2e.optical_layer, 32, 2, images_path, 'x')
    save_npy_metric(psnr_train, f'{metrics_path}/psnr_train')
    save_npy_metric(psnr_test, f'{metrics_path}/psnr_test')
    save_npy_metric(ssim_train, f'{metrics_path}/ssim_train')
    save_npy_metric(ssim_test, f'{metrics_path}/ssim_test')