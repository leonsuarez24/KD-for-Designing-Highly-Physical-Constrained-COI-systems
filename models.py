import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from psf_estimators import *
from fft_conv_pytorch import fft_conv

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
    def __init__(self, K, output_dim, input_dim):
        super(OpticalLayer, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.K = K
        ca = torch.normal(0, 1, (self.K, self.output_dim, self.input_dim))
        self.weights = nn.Parameter(ca / np.sqrt(self.output_dim * self.input_dim))

    def forward(self, x):
        # Downscale to 1 px
        y = self.forward_pass(x)
        # Upscale 
        x = self.transpose_pass(y)
        return x

    def forward_pass(self, x):
        ca_w = torch.unsqueeze(self.weights, 0)
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
    
class doe_layer(nn.Module):
    def __init__(self, Nz, Nw, Nx, Ny, Nu, Nv):
        super(doe_layer, self).__init__()
        self.heights = 1
        self.start_w = 400e-9
        self.end_w = 700e-9
        self.z_source = 50e-3 
        self.radii = 0.5e-3 
        self.focal_lens = 50e-3 
        self.du = 1e-3
        self.Ns = 3 
        self.start_z = 40e-3
        self.end_z = 65e-3
        self.Np = np.maximum(Nu, Nv)
        self.pitch = 1e-3*(1/self.Np)
        self.shift_y = int(self.Np*1)
        self.shift_x = int(self.Np*1)
        self.x_shiftings = np.linspace(-self.shift_x, self.shift_x, Nx)
        self.wavelengths = np.linspace(self.start_w, self.end_w, Nw)
        self.distances = np.linspace(self.start_z, self.end_z, Nz)
        self.y_shiftings = np.linspace(-self.shift_y, self.shift_y, Ny)
        self.Nf = self.Np*2

        ph = spiral_doe(self.start_w, 
                        self.end_w, 
                        self.Ns,
                        self.Np, 
                        self.radii, 
                        self.focal_lens, 
                        self.du)
        
        self.weights = nn.Parameter(torch.from_numpy(ph))


        def forward_pass(self, x, R, G, B):
            propa = calculate_psfs_doe(ph = self.weights.numpy(), 
                                       x_source = self.x_shiftings,
                                       y_source= self.y_shiftings, 
                                       z_source = self.z_source, 
                                       pitch = self.pitch,
                                       wavelengths = self.wavelengths,
                                       distances = self.distances,
                                       Nf = self.Nf)
            propa = propa.numpy()
            a = propa.reshape(Nz, Nw, Ny, Nx, -1)
            mina = a.min(axis=-1, keepdims=True).reshape(Nz, Nw, Ny, Nx,1, 1)
            maxa = a.max(axis=-1, keepdims=True).reshape(Nz, Nw, Ny, Nx,1, 1)

            propa = (propa - mina)/(maxa - mina)

            # -----------
            psfs = propa[0, :, 0, 0, :, :] 
            psfs_tensor = np.expand_dims(psfs, axis=1)
            psfs_tensor = torch.from_numpy(psfs_tensor)

            y = torch.zeros(x.shape)
            for i in range(x.shape[0]):
                y[i, 0, :, :] = fft_conv(x[i].unsqueeze(0), psfs_tensor[i].unsqueeze(0), padding='same')

            R_channel = torch.zeros((512, 512))
            G_channel = torch.zeros((512, 512))
            B_channel = torch.zeros((512, 512))

            for i in range(31):
                R_channel += y[i,0,:,:]*R[i]
                G_channel += y[i,0,:,:]*G[i]
                B_channel += y[i,0,:,:]*B[i]

            RGB = torch.zeros((512, 512, 3))
            RGB[:,:,0] = R_channel[:,:]/torch.max(R_channel)
            RGB[:,:,1] = G_channel[:,:]/torch.max(G_channel)
            RGB[:,:,2] = B_channel[:,:]/torch.max(B_channel)
            return RGB
        


class E2E_Unfolding_Base(nn.Module):
    def __init__(self, K, output_dim, input_dim, n_class, n_stages):
        super(E2E_Unfolding_Base, self).__init__()
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