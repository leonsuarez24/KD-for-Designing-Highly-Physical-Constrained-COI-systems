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
    def __init__(self, K, output_dim, input_dim, binary=True):
        super(OpticalLayer, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.binary = binary
        self.K = K
        ca = torch.normal(0, 1, (self.K, self.output_dim, self.input_dim))
        self.weights = nn.Parameter(ca / np.sqrt(self.output_dim * self.input_dim))

    def forward(self, x):
        # Downscale to 1 px
        y = self.forward_pass(x)
        # Upscale to 28x28
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
        ca_w = BinaryQuantize.apply(ca_w)
        x = torch.mul(y, ca_w)
        x = torch.sum(x, dim=1)
        x = torch.unsqueeze(x, 1)
        x = x / torch.max(x)
        return x

class E2E_Unfolding_Base(nn.Module):
    def __init__(self, K, output_dim, input_dim, n_class, n_stages, binary: bool):
        super(E2E_Unfolding_Base, self).__init__()
        self.optical_layer = OpticalLayer(K, output_dim, input_dim, binary)
        self.n_stages = n_stages
        self.proximals = nn.ModuleList(
            [Proximal_Mapping(channel=n_class).to('cuda')
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
    def __init__(self, K, output_dim, input_dim, n_class, n_stages):
        super(E2E_Unfolding_Distill, self).__init__()
        self.optical_layer = OpticalLayer(K, output_dim, input_dim)
        self.n_stages = n_stages
        self.proximals = nn.ModuleList(
            [Proximal_Mapping(channel=n_class).to('cuda')
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

