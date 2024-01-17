import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class OpticalLayer(nn.Module):
    def __init__(self, K, width, height, binary=True, snr:int = 20):
        super(OpticalLayer, self).__init__()
        self.width = width
        self.height = height
        self.binary = binary
        self.K = K
        ca = torch.normal(0, 1, (self.K, self.width, self.height))
        self.weights = nn.Parameter(ca / np.sqrt(self.width * self.height))
        self.snr = snr

    def forward(self, x):
        y = self.forward_pass(x) 
        w = self.noise(y)
        y = y + w
        x = self.transpose_pass(y)
        return x

    def forward_pass(self, x):
        ca_w = torch.unsqueeze(self.weights, 0)
        if self.binary:
            ca_w = BinaryQuantize.apply(ca_w)
        x = torch.mul(x, ca_w)
        x = torch.sum(x, dim=(-2, -1))
        x = torch.unsqueeze(x, -1)
        x = torch.unsqueeze(x, -1)
        return x

    def transpose_pass(self, y):
        ca_w = torch.unsqueeze(self.weights, 0)
        if self.binary:
            ca_w = BinaryQuantize.apply(ca_w)
        x = torch.mul(y, ca_w)
        x = torch.sum(x, dim=1)
        x = torch.unsqueeze(x, 1)
        x = x / torch.max(x)
        return x
    
    def noise(self, y):
        sigma = torch.sum(torch.pow(y, 2)) / ((y.shape[0] * y.shape[1]) * 10 ** (self.snr / 10))
        noise = torch.normal(mean=0, std=torch.sqrt(sigma).item(), size=y.shape)
        noise = noise.to(y.device)
        return noise

class E2E_Unfolding_Base(nn.Module):
    def __init__(self, K, width, height, channels, n_stages, binary: bool, snr:int):
        super(E2E_Unfolding_Base, self).__init__()
        self.optical_layer = OpticalLayer(K, width, height, binary, snr)
        self.n_stages = n_stages
        self.proximals = nn.ModuleList(
            [Proximal_Mapping(channel=channels).to('cuda')
             for _ in range(n_stages)
             ])
        self.alphas = nn.ParameterList(
            [
                nn.Parameter(torch.ones(1, requires_grad=True) * 0.01)
                for _ in range(n_stages)]
        )

        self.rhos = nn.ParameterList(
            [
                nn.Parameter(torch.ones(1, requires_grad=True) * 0.01)
                for _ in range(n_stages)]
        )

        self.betas = nn.ParameterList(
            [
                nn.Parameter(torch.ones(1, requires_grad=True) * 0.01)
                for _ in range(n_stages)]
        )

    def forward(self, x):
        y = self.optical_layer.forward_pass(x)
        x = self.optical_layer(x)
        u = torch.zeros_like(x)
        Xt = [x]
        for i in range(self.n_stages):
            # x = x - self.alphas[i]*(self.optical_layer.transpose_pass(self.optical_layer.forward_pass(x)-y) + self.rho[i]*(x-self.proximals[i](x)) )
            h, _ = self.proximals[i](x + u)
            x = x - self.alphas[i] * (
                    self.optical_layer.transpose_pass(self.optical_layer.forward_pass(x) - y) + self.rhos[i] * (
                    x - h + u))
            u = u + self.betas[i] * (x - h)
        return x


class E2E_Unfolding_Distill(nn.Module):
    def __init__(self, K, width, height, channels, n_stages, binary: bool):
        super(E2E_Unfolding_Distill, self).__init__()
        self.optical_layer = OpticalLayer(K, width, height, binary)
        self.n_stages = n_stages
        self.proximals = nn.ModuleList(
            [Proximal_Mapping(channel=channels).to('cuda')
             for _ in range(n_stages)
             ])
        self.alphas = nn.ParameterList(
            [
                nn.Parameter(torch.ones(1, requires_grad=True) * 0.01)
                for _ in range(n_stages)]
        )

        self.rhos = nn.ParameterList(
            [
                nn.Parameter(torch.ones(1, requires_grad=True) * 0.01)
                for _ in range(n_stages)]
        )

        self.betas = nn.ParameterList(
            [
                nn.Parameter(torch.ones(1, requires_grad=True) * 0.01)
                for _ in range(n_stages)]
        )

    def forward(self, x):
        y = self.optical_layer.forward_pass(x)
        x = self.optical_layer(x)
        u = torch.zeros_like(x)
        Xt = [x]
        Xs = []
        for i in range(self.n_stages):
            # x = x - self.alphas[i]*(self.optical_layer.transpose_pass(self.optical_layer.forward_pass(x)-y) + self.rho[i]*(x-self.proximals[i](x)) )
            h, xs = self.proximals[i](x + u)
            x = x - self.alphas[i] * (
                    self.optical_layer.transpose_pass(self.optical_layer.forward_pass(x) - y) + self.rhos[i] * (
                    x - h + u))
            u = u + self.betas[i] * (x - h)
            # x =
            Xt.append(x)
            Xs.append(xs)
        return Xt, Xs


class Proximal_Mapping(nn.Module):
    def __init__(self, channel):
        super(Proximal_Mapping, self).__init__()

        self.conv1 = nn.Conv2d(channel, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.theta = nn.Parameter(torch.ones(1, requires_grad=True) * 0.01).to('cuda')

        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(32, channel, kernel_size=3, padding=1)

        self.Sp = nn.Softplus()

    def forward(self, x):
        # Encode
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Softhreshold
        xs = torch.mul(torch.sign(x), F.relu(torch.abs(x) - self.theta))

        # Decode
        x = F.relu(self.conv5(xs))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))

        return x, xs

