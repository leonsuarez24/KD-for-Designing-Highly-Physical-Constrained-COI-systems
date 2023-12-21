import torch
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torchvision.transforms as transforms

def fft2(x, dim = (-2, -1)):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(x, dim), dim = dim), dim)

def ifft2(x, dim = (-2, -1)):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(x, dim), dim = dim), dim)

def ASM(wavefront, distance, wavelength, dx,  input_in_fourier=False):
    
    K = wavefront.shape[-1]
    wavelength = torch.from_numpy(wavelength).float()
    distance = torch.from_numpy(distance).float()

    # Frequency meshgrids
    k = torch.fft.fftfreq(K, d=dx)*2*np.pi; 
    # Center the frequency grid
    k = torch.fft.ifftshift(k)
    kx, ky = torch.meshgrid(k, k)

    s = [1]*(len(wavefront.shape)-2) + [wavefront.shape[-2], wavefront.shape[-1]]
    kx = kx.reshape(s)
    ky = ky.reshape(s)

    k0 = (2 * np.pi) / wavelength
    # Propagation kernel
    U = k0**2 - (kx**2 + ky**2)

    phase_mod = distance*torch.sqrt(U)


    length_x = len(k) * dx 

    # band-limited ASM - Matsushima et al. (2009)
    f_max = 2*np.pi / torch.sqrt((2 * distance* (1 / length_x) ) **2 + 1) / wavelength


    H_filter = torch.zeros_like(phase_mod)
    H_filter[ ( torch.abs(kx) < f_max) & (torch.abs(ky) < f_max) ] = 1
    H_filter = 1
    SFTF = torch.exp(1j*H_filter*phase_mod)*H_filter
    #SFTF[U<0] = 0

    if input_in_fourier:
        I_if = wavefront
    else:
        #I_if = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(wavefront)))
        I_if = fft2(wavefront)
    prop_wave = ifft2(I_if*SFTF)
    #prop_wave = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(I_if*SFTF)))
    
    return prop_wave

def deta(Lb):
    IdLens = 1.5375+0.00829045*Lb**(-2)-0.000211046*Lb**(-4);
    val = IdLens-1;
    return val

def numpy_pad(x, pad2):
    if type(x) != np.ndarray:
        x = np.array(x)
    if type(pad2) != np.array:
        pad2 = np.array(pad2)


    s = x.shape[-1]

    pad = (pad2-s)//2
    if pad >0:
        pad = [[0, 0]]*(len(x.shape)-2) + [[pad, pad], [pad, pad]]
        x = np.pad(x, pad, 'constant', constant_values=0)
    return x


def lamb_d(lm, lM, N, theta):
    val = 0.0
    #if theta >= 0 and theta < (2 * np.pi / N):
    val = lm + (lM - lm) * N * theta / 2 / np.pi
    #elif theta >= (2 * np.pi / N):
    #    val = lamb_d(lm, lM, N, theta - (2 * np.pi / N))
    return val

def spiral_doe(start_w, end_w, Ns, Np, radii, f, du):

    """
    Ns = Number of spirals
    Np = Number of pixels
    radii = Radius of the spiral
    f = Focal length of the lens
    start_w = Wavelength at the beging of the spiral
    end_w = Wavelength at the end of the spiral
    
    """
    #x = torch.linspace(-Np * du / 2, Np * du / 2, Np)
    x = torch.linspace(-du/2, du / 2, Np)
    Y, X = torch.meshgrid(x, x)
    theta, r = torch.arctan2(Y, X), torch.sqrt(X**2 + Y**2)

    theta += np.pi

    ph = torch.zeros((Np, Np))  # Initialize height map
    theta =  theta*(r<= radii)
    r = r*(r<= radii)
    theta = torch.remainder(theta, (2 * np.pi / Ns))
    lt = lamb_d(start_w, end_w, Ns, theta)  # Wavelength at theta

    n = torch.true_divide((torch.sqrt(r**2 + f**2) - f), lt)  # Constructive interference
    n = torch.ceil(n+1e-6)
    ph = (n * lt - (torch.sqrt(r**2 + f**2) - f)) / deta(lt * 1e6)  # Heights
    


    return ph.numpy()


def fresnell_lens(focal, wavelength, Np, radii, spacing = None):
    
    """
    Calculates the phase shift pattern for a Fresnel lens.

    Parameters:
    - focal (float): The focal length of the lens.
    - wavelength (float): The wavelength of the light.
    - Np (int): The number of points in the grid.
    - radii (float): The radius of the lens.
    - spacing (float, optional): The spacing between the phase shifts. If not provided, it is calculated based on the wavelength and refractive index.

    Returns:
    - z (ndarray): The phase shift pattern.

    """
    
    n = 0.5
    R = (focal*n)
    if spacing is None:
        spacing = wavelength/(n)
    else:
        spacing = spacing
    rad = radii/2
    x = np.linspace(-rad, rad, Np)
    [x, y] = np.meshgrid(x, x)
    z = np.sqrt(R**2 -( x**2 + y**2)/2)
    z = z * (x**2 + y**2 <= (rad)**2)
    z = np.mod(z,spacing)
    return z

    
def calc_one_psf(ph, x_source, y_source, z_source, pitch, wavelengths, distances, pad2 = 1000):


    n = ph.shape[-1]




    # # Frequency meshgrids
    k0 = (2 * np.pi) / wavelengths
    Ndelta = pitch 
    ks = np.linspace(-Ndelta, Ndelta, n)
    KX, KY = np.meshgrid(ks, ks)
    KX = KX.reshape(1, 1, n, n)
    KY = KY.reshape(1, 1, n, n)
    KX = KX/ wavelengths
    KY = KY/ wavelengths
    KZ = np.sqrt(k0**2 - (KX**2) - (KY**2))

    phase_doe = deta(wavelengths * 1e6) * k0 * ph


    shift_phase = np.exp(1j * (KX * x_source/KZ.shape[-2] + KY * y_source/KZ.shape[-1]))
    

    
    
    # Heigh map to phase map
    DOE = 1 * np.exp(1j * phase_doe)
    # F = Hlight * DOE


    Hlight = np.exp(-1j * z_source * KZ) * shift_phase


    x = Hlight*DOE

    x = numpy_pad(x, pad2)

    x = torch.tensor(x, dtype=torch.complex64)

    # propagation
    propa = np.abs(ASM(x, distances, wavelengths, pitch, input_in_fourier=False))**2

    

    return propa

def calculate_psfs_doe(ph, x_source, y_source, z_source, pitch, wavelengths, distances, Nf):

    """
    This function calculates the PSF of a spiral phase mask 
    """


    n_orig  = ph.shape[-1]

    psfs_full = torch.zeros((distances.shape[0], wavelengths.shape[0], x_source.shape[0], y_source.shape[0], n_orig,  n_orig))

    wavelengths = wavelengths.reshape(1, wavelengths.shape[0], 1, 1)
    ph = ph.reshape(1, 1, ph.shape[0], ph.shape[1])
    distances = distances.reshape(distances.shape[0], 1, 1, 1)




    #x_source = x_source.reshape(1, 1, 1, x_source.shape[0], 1, 1)
    #y_source = y_source.reshape(1, 1, y_source.shape[0], 1, 1, 1)
    if x_source.shape[0] == 1:
        x_source[0] = 0
    if y_source.shape[0] == 1:
        y_source[0] = 0



    result_pad = (Nf-n_orig)//2
    for xshift in range(x_source.shape[0]):
        for yshift in range(y_source.shape[0]):
            res = calc_one_psf(ph, x_source[xshift], y_source[yshift], z_source, pitch, wavelengths, distances, pad2=Nf)
            res = (res - res.min())/(res.max() - res.min())
            # Delete the padding
            psfs_full[:, :, yshift, xshift, ...] = res[:, :, result_pad:-result_pad, result_pad:-result_pad]

    return psfs_full


def get_dataset_doe(which_doe, Nz, Nw, Nx, Ny, Nu, Nv):
    """
    Function to generate the dataset of PSFs for a given DOE
        args:
            which_doe: str, 'spiral', 'fresnel', 'custom'
            Nz: int, number of Depths
            Nw: int, number of Wavelengths
            Nx: int, number of X shifts
            Ny: int, number of Y shifts
            Nu: int, number of pixels in X
            Nv: int, number of pixels in Y
        return:
            propa: torch.tensor, PSFs of size Nz x Nw x Ny x Nx
            wavelengths: np.array, wavelengths used to generate the PSFs
            distances: np.array, distances used to generate the PSFs
    """
    # Wavelegth range
    start_w =  400e-9
    end_w = 700e-9

    Np = np.maximum(Nu, Nv)
    shift_y = int(Np*1)
    shift_x = int(Np*1)

    if which_doe == 'spiral':
        z_source = 50e-3 # distance from the DOE to the sensor
        radii = 0.5e-3 # radius of the spiral
        focal_lens = 50e-3 # focal length of the spiral lens
        du = 1e-3; # pixel pitch of Spiral DOE
        Ns = 3 # number of spirals
        start_z = 40e-3
        end_z = 65e-3

        # Parameters default
        pitch = 1e-3*(1/Np)#4e-6 
        
    elif which_doe == "fresnel":
        focal_lens = 200e-3 # 200 mm 
        z_source = 200e-3
        radii = 5e-3 # radius of the spiral
        design_lambda = 550e-9
        #start_z = 13.5e-3
        #end_z = start_z + 6e-3#23e-3
        start_z = 150e-3
        end_z = start_z + 100e-3#23e-3
        pitch = 3.6e-3*(1/Np)##
        #pitch = 1e-6
    elif which_doe == "custom":
        focal_lens = 200e-3 # 200 mm 
        z_source = 200e-3
        radii = 5e-3 # radius of the spiral
        design_lambda = 550e-9
        #start_z = 13.5e-3
        #end_z = start_z + 6e-3#23e-3
        start_z = 150e-3
        end_z = start_z + 100e-3#23e-3
        pitch = 3.6e-3*(1/Np)##
    else:
        raise ValueError('which_doe should be spiral or fresnel')
    
    Nf = Np*2
    wavelengths = np.linspace(start_w, end_w, Nw)
    distances = np.linspace(start_z, end_z, Nz)
    y_shiftings = np.linspace(-shift_y, shift_y, Ny)
    x_shiftings = np.linspace(-shift_x, shift_x, Nx)



    if distances.shape[0] == 1:
        distances[0] = (start_z+end_z)/2

    
    if wavelengths.shape[0] == 1:
        wavelengths[0] = (start_w+end_w)/2

    if which_doe == 'spiral':
        ph = spiral_doe(start_w, end_w, Ns, Np, radii, focal_lens, du)
    elif which_doe == "fresnel":
        ph = fresnell_lens(focal = focal_lens, wavelength = design_lambda, Np = Np, radii = radii)
    elif which_doe == "custom":
        ## ------------------ Load the custom DOE ----------------------------# You can check how to generate a custom DOE based on fresnel and spiral DOEs
        ph = loadmat('custom_doe.mat')['ph'] # Here load the custom DOE. The DOE should be a Np x Np matrix
    else:
        raise ValueError('which_doe should be spiral or fresnel')

    # Calculate the PSFs
    propa = calculate_psfs_doe(ph = ph, x_source = x_shiftings, y_source= y_shiftings, z_source = z_source, pitch = pitch,
                               wavelengths = wavelengths, distances = distances, Nf = Nf)

    propa = propa.numpy()
    a = propa.reshape(Nz, Nw, Ny, Nx, -1)
    mina = a.min(axis=-1, keepdims=True).reshape(Nz, Nw, Ny, Nx,1, 1)
    maxa = a.max(axis=-1, keepdims=True).reshape(Nz, Nw, Ny, Nx,1, 1)

    propa = (propa - mina)/(maxa - mina)
  
    return propa, wavelengths, distances


