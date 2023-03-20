import torch
import torch.nn as nn
import numpy as np
from fastai.vision import ImageSegment, PixelShuffle_ICNR
from fastai.layers import NormType
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.distributions import Normal
from turtle import forward
from fastai.vision import conv_layer
from fastai.vision.data import SegmentationLabelList, SegmentationProcessor, SegmentationItemList
from fastai.vision.transform import get_transforms
from fastai.vision.image import open_image, open_mask, Image
from pathlib import Path

l1_loss = torch.nn.L1Loss()


class Block(nn.Module):
    """Basic convolutional building block

    Parameters
    ----------
    in_ch : int
        number of input channels to the block
    out_ch : int     
        number of output channels of the block
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu =  nn.LeakyReLU() # TODO  # leaky ReLU
        self.bn1 = nn.BatchNorm2d(out_ch)# TODO   # batch normalisation or in_ch
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch) # TODO out_ch = num_features?

    def forward(self, x):
        """Performs a forward pass of the block
       
        x : torch.Tensor
            the input to the block
        torch.Tensor
            the output of the forward pass
        """
        # a block consists of two convolutional layers
        # with ReLU activations
        # use batch normalisation

        # TODO
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    



class SpadeBN(nn.Module):
    def __init__(self, nf):
        super(SpadeBN, self).__init__()
        self.bn = nn.BatchNorm2d(nf, affine=False)
        self.conv0 = conv_layer(1,128, norm_type=NormType.Spectral)
        self.conv1 = conv_layer(128, nf, norm_type=NormType.Spectral)
        self.conv2 = conv_layer(128, nf, norm_type=NormType.Spectral)
    
    def forward(self, features, mask):
        size = features.size()[-2:]
        mask = F.interpolate(mask.float(), size=size)
        interim_conv = self.conv0(mask)
        gamma = self.conv1(interim_conv)
        beta = self.conv2(interim_conv)
        return ((self.bn(features) * gamma) + beta)



# SPADE ResBlock
class SpadeResBlock(nn.Module):
    def __init__(self, ni, nf):
        super(SpadeResBlock, self).__init__()
        self.spade_bn0 = SpadeBN(ni)
        self.conv0 = conv_layer(ni, nf, use_activ=False, norm_type=NormType.Spectral)
        self.spade_bn1 = SpadeBN(nf)
        self.conv1 = conv_layer(nf, nf, use_activ=False, norm_type=NormType.Spectral)
        self.spade_skip = SpadeBN(ni)
        self.conv_skip = conv_layer(ni, nf, use_activ=False, norm_type=NormType.Spectral, )
    
    def forward(self, features, mask):
        skip_features = self.conv_skip(F.relu(self.spade_skip(features, mask)))
        features = self.conv0(F.relu(self.spade_bn0(features, mask)))
        features = self.conv1(F.relu(self.spade_bn1(features, mask)))
        return skip_features + features

class Encoder(nn.Module):
    """The encoder part of the VAE.

    Parameters
    ----------
    spatial_size : list[int]
        size of the input image, by default [64, 64]
    z_dim : int
        dimension of the latent space
    chs : tuple
        hold the number of input channels for each encoder block
    """

    def __init__(self, spatial_size=[64, 64], z_dim=256, chs=(1, 64, 128, 256)):
        super().__init__()
        # convolutional blocks
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]# TODO = range 0,1,2,3
        )
        
        # max pooling
        self.pool = nn.MaxPool2d(2)# TODO
        # height and width of images at lowest resolution level
        _h, _w = 8, 8 #(int(np.sqrt(spatial_size[0])), int(np.sqrt(spatial_size[0]))) #spatial_size # TODO

        # flattening
        self.out = nn.Sequential(nn.Flatten(1), nn.Linear(chs[-1] * _h * _w, 2 * z_dim))

    def forward(self, x):
        """Performs the forward pass for all blocks in the encoder.

        Parameters
        ----------
        x : torch.Tensor
            input image to the encoder

        Returns
        -------
        list[torch.Tensor]    
            a tensor with the means and a tensor with the log variances of the
            latent distribution
        """

        for block in self.enc_blocks:
            x = block(x) # forward pass of block (conv block) # TODO: conv block  
            x = self.pool(x) # TODO: pooling 
        x = self.out(x)
      #  x = nn.Tanh() # TODO: output layer  
        return torch.chunk(x, 2, dim=1)  # 2 chunks, 1 each for mu and logvar


class Generator(nn.Module):
    """Generator of the GAN

    Parameters
    ----------
    z_dim : int 
        dimension of latent space
    chs : tuple
        holds the number of channels for each block
    h : int, optional
        height of image at lowest resolution level, by default 8
    w : int, optional
        width of image at lowest resolution level, by default 8    
    """

    def __init__(self, z_dim=256, chs=(256, 128, 64, 32), h=8, w=8):

        super().__init__()
        self.chs = chs
        self.h = h  
        self.w = w  
        self.z_dim = z_dim  
        self.proj_z = nn.Linear(
            self.z_dim, self.chs[0] * self.h * self.w
        )  # fully connected layer on latent space
        self.reshape = lambda x: torch.reshape(
            x, (-1, self.chs[0], self.h, self.w)
        )  # reshaping

        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i], chs[i], 2, 2) for i in range(len(chs) - 1)] # TODO: transposed convolution 
        )

        self.dec_blocks = nn.ModuleList(
            [nn.Conv2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)]
          #  [Block(2 * chs[i], chs[i + 1]) for i in range(len(chs) - 1)] # TODO: conv block           
        )

    def forward(self, z):
        """Performs the forward pass of decoder

        Parameters
        ----------
        z : torch.Tensor
            input to the generator
        
        Returns
        -------
        x : torch.Tensor
        
        """
        # nn.Linear(input_size, dim)
    #    input_size = self.h*self.w
        x = self.proj_z(z) # TODO: fully connected layer
    #    x = nn.Linear(input_size, self.z_dim*self.h*self.w) # TODO: fully connected layer input: n_features, out_features
        x = self.reshape(x)# TODO: reshape to image dimensions , torch.tensor(self.h, self.w)
        
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
          #  upconv = self.upconvs(x)[i:] # TODO: transposed convolution
            x = self.dec_blocks[i](x) # TODO: convolutional block
      
    #    x = nn.Tanh(nn.ConvTranspose2d(self.chs[-1], 16, 1, 2))
    #    x = nn.Tanh(nn.Linear(x)) #  self.proj_o(conv) # TODO: output layer
        return x



# SpadeGenereator
class UpsamplingNearest2d(nn.Module):
    def forward(self, x):
        return F.interpolate(x, recompute_scale_factor=2)


nfs = [1024,1024,512,256,128,64]
scale = [2] * len(nfs)
input_noise_dim = 256

class SpadeGenerator(nn.Module):
    def __init__(self, input_noise_dim=input_noise_dim,
                 nfs=nfs,
                 pixel_shuffle_upsampling=True,
                 batch_size = batch_size,
                 input_image=None):
                 
        super(SpadeGenerator, self).__init__()
        self.input_noise_dim = input_noise_dim
        self.batch_size = batch_size
        self.linear = nn.Linear(input_noise_dim, 16384)
        self.spade_upsample_blocks = nn.ModuleList([])
        for i in range(len(nfs)):
            self.spade_upsample_blocks.append(nn.ModuleList([SpadeResBlock(1024 if i == 0 else nfs[i-1], nfs[i]),
                           PixelShuffle_ICNR(nfs[i], nfs[i], scale=2, norm_type=NormType.Spectral) if pixel_shuffle_upsampling else UpsamplingNearest2d()]))
        self.conv_final = conv_layer(nfs[-1], 3, use_activ=False, bias=True, norm_type=None)

    def forward(self.mask):
        gaussian_noise = Normal(0,1).sample(torch.Size([mask.size()[0], self.input_noise_dim])).to(device_).detach()
        linear_features = self.linear(gaussian_noise)
        spatial_features = linear_features.view(self.batch_size, 1024, 4, 4) # hardcoded
        for block in self.spade_upsample_blocks:
            spatial_features = block[0](spatial_features, mask)
            spatial_features = block[1](spatial_features)
#         pdb.set_trace()
        return (torch.tanh(self.conv_final(spatial_features)) + 1)/2


class VAE(nn.Module):
    """A representation of the VAE

    Parameters
    ----------
    enc_chs : tuple 
        holds the number of input channels of each block in the encoder
    dec_chs : tuple 
        holds the number of input channels of each block in the decoder
    """
    def __init__(
        self,
        enc_chs=(1, 64, 128, 256),
        dec_chs=(256, 128, 64, 32),
    ):
        super().__init__()
        self.encoder = Encoder()
        self.SpadeGenarator = SpadeGenerator()
     #   self.head = nn.Sequential(nn.Conv2d(dec_chs[-1], 1, 1,2,1), nn.Tanh())# tanh activation)
        self.head = nn.Sequential(nn.ConvTranspose2d(dec_chs[-1], 1, 38, 4, 1), nn.Tanh())
    #    self.head = nn.Sequential(nn.Linear(2048, ))

    def forward(self, x):
        """Performs a forwards pass of the VAE and returns the reconstruction
        and mean + logvar.

        Parameters
        ----------
        x : torch.Tensor
            the input to the encoder

        Returns
        -------
        torch.Tensor
            the reconstruction of the input image
        float
            the mean of the latent distribution
        float
            the log of the variance of the latent distribution
        """
        mu, logvar = self.encoder(x)
        latent_z = sample_z(mu, logvar)
        
        recon = self.generator(latent_z)
        output = self.head(recon)

        return output, mu, logvar


def get_noise(n_samples, z_dim, device="cpu"):
    """Creates noise vectors.
    
    Given the dimensions (n_samples, z_dim), creates a tensor of that shape filled with 
    random numbers from the normal distribution.

    Parameters
    ----------
    n_samples : int
        the number of samples to generate
    z_dim : int
        the dimension of the noise vector
    device : str
        the type of the device, by default "cpu"
    """
    return torch.randn(n_samples, z_dim, device=device)


def sample_z(mu, logvar):
    """Samples noise vector from a Gaussian distribution with reparameterization trick.

    Parameters
    ----------
    mu : float
        the mean of the distribution
    logvar : float
        the log of the variance of the distribution
    """
    eps = torch.randn(mu.size(), device=mu.device).to(mu.dtype)
    return (logvar / 2).exp() * eps + mu


def kld_loss(mu, logvar):
    """Computes the KLD loss given parameters of the predicted 
    latent distribution.

    Parameters
    ----------
    mu : float
        the mean of the distribution
    logvar : float
        the log of the variance of the distribution

    Returns
    -------
    float
        the kld loss

    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def vae_loss(inputs, recons, mu, logvar):
    """Computes the VAE loss, sum of reconstruction and KLD loss

    Parameters
    ----------
    inputs : torch.Tensor
        the input images to the vae
    recons : torch.Tensor
        the predicted reconstructions from the vae
    mu : float
        the predicted mean of the latent distribution
    logvar : float
        the predicted log of the variance of the latent distribution

    Returns
    -------
    float
        sum of reconstruction and KLD loss
    """
    print(recons.size())
    return l1_loss(inputs, recons) + kld_loss(mu, logvar)

