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
    def __init__(self, K, width, height):
        super(OpticalLayer, self).__init__()
        self.width = width
        self.height = height
        self.K = K
        ca = torch.normal(0, 1, (self.K, self.width, self.height))
        self.weights = nn.Parameter(ca / np.sqrt(self.width * self.height))

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
        self.Nz = Nz
        self.Nw = Nw
        self.Nx = Nx
        self.Ny = Ny
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
        self.resnet_layer = Resnet(Resnet_Block, [4,4,4,4])


        self.ph = spiral_doe(self.start_w, 
                        self.end_w, 
                        self.Ns,
                        self.Np, 
                        self.radii, 
                        self.focal_lens, 
                        self.du)
        
        self.weights = nn.Parameter(torch.from_numpy(self.ph))
        self.deconv = nn.ConvTranspose2d(1, 1, 31, padding = 255, dilation=1, bias = False)

    def forward(self, x, R, G, B):
        out = self.forward_pass(x, R, G, B)
        out = self.backward_pass(out)
        return out


    def forward_pass(self, x, R, G, B):
        propa = calculate_psfs_doe(ph = self.weights.detach().clone().numpy(), 
                                    x_source = self.x_shiftings,
                                    y_source= self.y_shiftings, 
                                    z_source = self.z_source, 
                                    pitch = self.pitch,
                                    wavelengths = self.wavelengths,
                                    distances = self.distances,
                                    Nf = self.Nf)
        propa = propa.numpy()
        a = propa.reshape(self.Nz, self.Nw, self.Ny, self.Nx, -1)
        mina = a.min(axis=-1, keepdims=True).reshape(self.Nz, self.Nw, self.Ny, self.Nx,1, 1)
        maxa = a.max(axis=-1, keepdims=True).reshape(self.Nz, self.Nw, self.Ny, self.Nx,1, 1)

        propa = (propa - mina)/(maxa - mina)

        # -----------
        psfs = propa[0, :, 0, 0, :, :] 
        psfs_tensor = np.expand_dims(psfs, axis=1)
        psfs_tensor = torch.from_numpy(psfs_tensor)

        y = torch.zeros(x.shape)
        for j in range(x.shape[0]):     # Loops in batch size
            for i in range(x.shape[1]): # Loops in wavelengths
                y[j, i, :, :] = fft_conv(x[j,i,:,:].unsqueeze(0).unsqueeze(0), psfs_tensor[i,:,:].unsqueeze(0), padding='same')

        for j in range(x.shape[0]):     # Loops in batch size

            R_channel = torch.zeros((512, 512))
            G_channel = torch.zeros((512, 512))
            B_channel = torch.zeros((512, 512))

            for i in range(31):         # Loops in wavelengths
                R_channel += y[j,i,:,:]*R[i]
                G_channel += y[j,i,:,:]*G[i]
                B_channel += y[j,i,:,:]*B[i]

            RGB = torch.zeros((x.shape[0], 3, 512, 512))
            RGB[j,0,:,:] = R_channel[:,:]/torch.max(R_channel)
            RGB[j,1,:,:] = G_channel[:,:]/torch.max(G_channel)
            RGB[j,2,:,:] = B_channel[:,:]/torch.max(B_channel)
            return RGB
    
    def backward_pass(self, RGB):
        out = self.resnet_layer(RGB)
        return out


class E2E_Unfolding_Base(nn.Module):
    def __init__(self,                 
                 K, 
                 width, 
                 height, 
                 channels, 
                 n_stages, 
                 doe:bool, 
                 Nz, Nw, Nx, Ny, Nu, Nv):
        super(E2E_Unfolding_Base, self).__init__()

        if doe:
            self.optical_layer = doe_layer(Nz, Nw, Nx, Ny, Nu, Nv)
        else:
            self.optical_layer = OpticalLayer(K, width, height)
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
    def __init__(self, 
                 K, 
                 width, 
                 height, 
                 channels, 
                 n_stages, 
                 doe:bool, 
                 Nz, Nw, Nx, Ny, Nu, Nv):
        super(E2E_Unfolding_Distill, self).__init__()
        if doe:
            self.optical_layer = doe_layer(Nz, Nw, Nx, Ny, Nu, Nv)
        else:
            self.optical_layer = OpticalLayer(K, width, height)
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

class Resnet_Block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample = None):
        super().__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, 
                                  out_channels, 
                                  kernel_size = 3, 
                                  stride = 1, 
                                  padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, 
                                  out_channels,     
                                  kernel_size = 3, 
                                  stride = 1, 
                                  padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
        self.downsample = downsample
        
    def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.conv2(out)
            if self.downsample:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out

class Resnet(nn.Module):
    def __init__(self, block, num_layers):
        super().__init__()
        self.inplanes = 3
        self.layer0 = self._make_layer(block, 10, num_layers[0])
        self.layer1 = self._make_layer(block, 17, num_layers[1])
        self.layer2 = self._make_layer(block, 24, num_layers[2])
        self.layer3 = self._make_layer(block, 31, num_layers[3])

        
    def _make_layer(self, block, planes, num_layers):
        downsample = None
        if self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=3, stride=1, padding = 1),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, downsample))
        self.inplanes = planes
        for i in range(1, num_layers):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
