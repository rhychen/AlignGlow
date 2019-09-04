# -*- coding: utf-8 -*-
"""
Pytorch implementation of [1].

[1] D. P. Kingma, et al. "Glow: Generative Flow with Invertible 1x1 Convolutions", https://arxiv.org/abs/1807.03039
[2] L. Dinh, et al. "Density estimation using Real NVP", https://arxiv.org/abs/1605.08803
[3] Nalisnick et al. “Do Deep Generative Models Know what They Don’t Know?”, ICLR 2019. https://arxiv.org/abs/1810.09136
[4] Han Zhang, et. al., "Self-Attention Generative Adversarial Networks." arXiv preprint arXiv:1805.08318 (2018)
[5] Wang, Xiaolong, et. al. "Non-local neural networks." CVPR, 2018.
"""

import sys
sys.path.append(r'C:\AI, Machine learning')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.datasets as dset

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os, os.path
import zipfile
import math
import time
import pickle
import random
from datetime import datetime
from collections import namedtuple
from collections import defaultdict
from PIL import Image

from utility import *

print(torch.__version__)

debug       = False
in_colab    = False
new_session = True
colour_img  = False

if debug: 
    print("\n!!! RUNNING DEBUG MODE !!!\n")
    # Fix seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    debug_dict = {}

###############################
# Google Colab setup
###############################
if in_colab:
    from google.colab import widgets
    #from google.colab import drive
    
    # Google Drive is mounted as 'gdrive'
    drive_path = '/content/gdrive/My Drive/'
    data_path  = drive_path
    grid       = widgets.Grid(2, 1)
    
    num_workers = 10
    
    # Mount Google Drive
    #from google.colab import drive
    #drive.mount(drive_path)
    
    # Download from Colab to local drive
    #from google.colab import files
    #files.download(<filename>)
else:
    drive_path = 'C:/AI, Machine learning/'
    data_path  = 'C:/datasets/'
    # Set to 0 to avoid issue with multiprocessing on Windows
    # https://discuss.pytorch.org/t/brokenpipeerror-errno-32-broken-pipe-when-i-run-cifar10-tutorial-py/6224/3
    num_workers = 0

###############################
# Misc setup
###############################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

timestamp = datetime.now()
timestr   = timestamp.strftime("%d") + timestamp.strftime("%m") +\
            timestamp.strftime("%H") + timestamp.strftime("%M")

# Results & logs directory
output_dir = 'AlignGlow_out_' + timestr
if in_colab:
    output_dir = drive_path + output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Saved model directory
chkpt_dir = 'AlignGlow_chkpt_' + timestr
if in_colab:
    chkpt_dir = drive_path + chkpt_dir
if not os.path.exists(chkpt_dir):
    os.makedirs(chkpt_dir)

#log_fh   = open('{}/training.log'.format(output_dir), 'w', buffering=1)
#tplot_fh = open('{}/training_plot.csv'.format(output_dir), 'w', buffering=1)
#vplot_fh = open('{}/validation_plot.csv'.format(output_dir), 'w', buffering=1)
#tplot_fh.write('# total\n')
#vplot_fh.write('# total\n')

Likelihood = namedtuple("Likelihood", ["NLL"])

# Option to carry on training from a previous session
if not new_session:
    saved_model_path = 'AlignGlow_21060625/epoch_10.pth'
    if in_colab:
        saved_model_path = drive_path + saved_model_path
    print("WARNING: Continuing from saved model {}".format(saved_model_path))
    saved_states = torch.load(saved_model_path, map_location=device)

# Number of new samples to generate
num_new_samples = 25

train_objective = Likelihood([]) if new_session else saved_states['train_objective']
val_objective   = Likelihood([]) if new_session else saved_states['val_objective']

###############################
# Hyperparameters
###############################

num_epochs  = 100
start_epoch = 1 if new_session else saved_states['epoch']
batch_size  = 32 if new_session else saved_states['hyperparams']['batch_size']
lr          = 3e-4 if new_session else saved_states['hyperparams']['lr']

# Levels in multi-scale model
model_lvl  = 3 if new_session else saved_states['hyperparams']['model_lvl']
# Flow steps per level
flow_steps = 10 if new_session else saved_states['hyperparams']['flow_steps']

###############################
# Dataset
###############################

#https://discuss.pytorch.org/t/beginner-how-do-i-write-a-custom-dataset-that-allows-me-to-return-image-and-its-target-image-not-label-as-a-pair/13988
#https://discuss.pytorch.org/t/upload-a-customize-data-set-for-multi-regression-task/43413
class PairedDataset(dset.DatasetFolder):
    """
     
    """
    def __init__(self, root, root2, transform=None, loader=pil_l_loader):
        self.root      = root
        self.root2     = root2
        self.transform = transform
        self.loader    = loader
        
        classes, class_to_idx = self._find_classes(self.root)
                
        # Second image set
        dir2     = os.path.expanduser(self.root2)
        img_set2 = defaultdict(list)
        for target in sorted(classes):
            d2 = os.path.join(dir2, target)
            if not os.path.isdir(d2):
                print("WARNING: Skipping {}; not a directory.".format(d2))
                continue
            for path, _, fnames in sorted(os.walk(d2)):
                for fname in fnames:
                    fullpath = os.path.join(path, fname)
                    img_set2[target].append(fullpath)
        
        # First image set
        dir1         = os.path.expanduser(self.root)
        self.samples = []
        for target in sorted(classes):
            d1 = os.path.join(dir1, target)
            if not os.path.isdir(d1):
                print("WARNING: Skipping {}; not a directory.".format(d1))
                continue
            for path, _, fnames in sorted(os.walk(d1)):
                for fname in sorted(fnames):
                    path1 = os.path.join(path, fname)
                    path2 = random.choice(img_set2[target])
                    item  = (path1, path2, target)
                    self.samples.append(item)
        
    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset. Copied from DatasetFolder class.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path1, path2, target = self.samples[index]
#        print('path1:', path1)
#        print('path2:', path2)
        img1 = self.loader(path1)
        img2 = self.loader(path2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return (img1, img2, target)

# Pixel quantization, e.g. CIFAR pixel values are 0 ~ 255
quan_lvl = 256

# De-quantize and shift to range of [-0.5, 0.5].
# Note ToTensor() converts a PIL Image in the range [0, 255] to tensor
# in the range [0.0, 1.0].
dequan = T.Compose([T.ToTensor(),
                    lambda x: x + torch.rand_like(x) / quan_lvl - 0.5])

if in_colab:
    archive_path = data_path + 'kkanji2_32x32_split.zip'
    extract_path = './'
    train_data   = extract_path + 'kkanji2_32x32_split/train'
    val_data     = extract_path + 'kkanji2_32x32_split/validation'
    zip_ref      = zipfile.ZipFile(archive_path, 'r')
    zip_ref.extractall(extract_path)
    zip_ref.close()
    archive_path = data_path + "curated_32x32_rotate.zip"
    extract_path = './'
    root2        = extract_path + "curated_32x32_rotate"
    zip_ref      = zipfile.ZipFile(archive_path, 'r')
    zip_ref.extractall(extract_path)
    zip_ref.close()
else:
    train_data = data_path + "kkanji2_32x32_split/train"
    val_data   = data_path + "kkanji2_32x32_split/validation"
    root2      = data_path + "curated_32x32_rotate"

# Training set
train_dataset = PairedDataset(root=train_data,
                              root2=root2,
                              transform=dequan,
                              loader=pil_l_loader,
                             )
train_loader  = DataLoader(train_dataset,
                           batch_size=batch_size,
                           shuffle=(not debug),
                           num_workers=num_workers)

# Validation set
val_dataset = PairedDataset(root=val_data,
                            root2=root2,
                            transform=dequan,
                            loader=pil_l_loader,
                           )
val_loader  = DataLoader(val_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=num_workers)

# Image dimensions
train_iter        = iter(train_loader)
img1, img2, label = train_iter.next()
_, C, H, W        = img1.size()
_, C2, H2, W2     = img2.size()
img_dim           = C * H * W
assert C == C2 and H == H2 and W == W2
print("H, W =", H, W)

###############################
# Model
###############################

#---------------------
# Convolution layers
#---------------------
'''
Convolution followed by ActNorm
'''
class Conv2dActNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        padding      = (kernel_size - 1) // 2 or padding
        self.conv    = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.actnorm = ActNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.actnorm.forward_flow(x, -1)[0]
        return x

'''
Convolution Layer with zero initialisation
'''
class Conv2dZeroInit(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, logs_factor=3):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.logs_factor = logs_factor
        self.logscale    = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)
        # Above is equivalent to:
        #with torch.no_grad():
        #    self.weight.zero_()
        #    self.bias.zero_()

    def forward(self, in_tensor):
        out_tensor = super().forward(in_tensor)
        if debug:
            self.weight.register_hook(check_backprop('Conv2dZeroInit W'))
            out_tensor.register_hook(check_backprop('Conv2dZeroInit out_tensor'))
        return out_tensor * torch.exp(self.logscale * self.logs_factor)

#---------------------
# Self-attention
#---------------------

# We follow the naming convention in [4]
class SelfAttn(nn.Module):
    def __init__(self, in_channels, lower_tri):
        super().__init__()
        
        self.lower_tri = lower_tri
        self.gamma     = nn.Parameter(torch.zeros(1))

        # 1x1 convolution layers. [4] uses out_channels = in_channels / 8
        # for f and g.
        scale = 1 if (in_channels < 8) else 8
        self.conv_f = nn.Conv2d(in_channels, in_channels // scale, kernel_size=1)
        self.conv_g = nn.Conv2d(in_channels, in_channels // scale, kernel_size=1)
        self.conv_h = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward_flow(self, x, lower_tri=None):
        if lower_tri is None:
            lower_tri = self.lower_tri
        
        B, C, H, W = x.size()
        HW = H * W
        
        f = self.conv_f(x)  # Query
        g = self.conv_g(x)  # Key
        h = self.conv_h(x)  # Value
        
        # Ignore batch dimension in the following comments since it's not important.
        # Reshape to 2D matrices (see [5], Fig. 2) by flattening height & width
        f = f.view(B, -1, HW)
        g = g.view(B, -1, HW)
        h = h.view(B, -1, HW)
        
        # Attention map, b
        # Each row of b sums to 1, so each element represents the relative amount
        # of attention (total attention being 1). I.e. b[i, j] is the amout of
        # attention the model pays to the j-th location when synthesizing the
        # i-th position.
        s = torch.matmul(f.transpose(1, 2), g)  # Tensor size: (H*W, H*W)
        # REVISIT: Try the dot-product alternative - Eq. (4) in [5]
        b = F.softmax(s, dim=1)                 # Each row sums to 1        

        # To ensure a triangular Jacobian, ouput[:, j] can only be dependent on 
        # input[:, 0:j-1] (for lower triangular) or [:, j:H*W] (for upper triangular).
        # This can be done by simply masking the attention map with a triangular mask.
        # By alternating beteen upper and lower triangular mask, every output point
        # has a chance to mix with every input point.
        if lower_tri:
            tri_mask = torch.tril(torch.ones_like(b))
            b_masked = b.transpose(1, 2) * tri_mask
        else:
            tri_mask = torch.triu(torch.ones_like(b))
            b_masked = b.transpose(1, 2) * tri_mask
        
        o = torch.matmul(h, b_masked).view(B, C, H, W)
        
        # REVISIT: Try introducing a parameter to scale the residual connection
        # REVISIT: so the model can learn to reduce the residual contribution to
        # REVISIT: 0 if needed (initialise the scale parameter to 1). Perhaps try
        # REVISIT: y = self.gamma * o + (1 - self.gamma) * x
        # Non-local block: non-local op (attention) + residual connection [5]
        y = self.gamma * o + x
        
        return y
    
    def reverse_flow(self, y):
        
        return x
    
    def forward(self, in_tensor, logdet, direction):
        if (direction == 'forward'):
            return self.forward_flow(in_tensor, logdet)
        else:
            return self.reverse_flow(in_tensor)
    
#---------------------
# Gaussian prior
#---------------------
'''
Compute the log-likelihood in forward flow assuming a Gaussian prior for the
latent variable.
'''
class GaussianPrior(nn.Module):
    def __init__(self, input_shape, mode=None):
        super().__init__()
        
        self.mode = mode
        self.C, self.H, self.W = input_shape
        
        self.const = -0.5 * math.log(2 * math.pi)
        
        # [1] uses class conditioned prior for CIFAR dataset
        if (mode == 'class_cond'):
            # REVISIT
            pass
        elif (mode == 'trainable'):
            self.conv = Conv2dZeroInit(2 * self.C, 2 * self.C, 3, padding=1)
        
    def forward_flow(self, fx, logdet):       
        mean  = torch.zeros_like(fx, dtype=torch.float, device=device)
        logsd = torch.zeros_like(fx, dtype=torch.float, device=device)
        
        if (self.mode == 'class_cond'):
            # REVISIT
            pass
        elif (self.mode == 'trainable'):
            mean_and_logsd = torch.cat((mean, logsd), dim=1)
            mean_and_logsd = self.conv(mean_and_logsd)
            mean, logsd    = torch.chunk(mean_and_logsd, 2, dim=1)

        sd     = torch.exp(logsd)
        log_pz = self.const - logsd - 0.5 * ((fx - mean) / sd) ** 2
        log_px = torch.sum(log_pz, dim=(1, 2, 3)) + logdet
        
        if debug:
            log_px.register_hook(check_backprop('GaussianPrior log_px'))
            fx.register_hook(check_backprop('GaussianPrior fx'))
        
        return fx, log_px
        
    def reverse_flow(self, z):
        # Sample from the prior distribution to generate new samples
        if z is None:
            mean  = torch.zeros(num_new_samples, self.C, self.H, self.W, dtype=torch.float, device=device)
            logsd = torch.zeros(num_new_samples, self.C, self.H, self.W, dtype=torch.float, device=device)
            
            if (self.mode == 'class_cond'):
                # REVISIT
                pass
            elif (self.mode == 'trainable'):
                mean_and_logsd = torch.cat((mean, logsd), dim=1)
                mean_and_logsd = self.conv(mean_and_logsd)
                mean, logsd    = torch.chunk(mean_and_logsd, 2, dim=1)

            sd  = torch.exp(logsd)
            eps = torch.randn_like(sd)
            z   = eps.mul(sd).add(mean)

        return z
        
    def forward(self, in_tensor, logdet, direction):
        if (direction == 'forward'):
            return self.forward_flow(in_tensor, logdet)
        else:
            return self.reverse_flow(in_tensor)

'''
As in [2], we factor out and Gaussianize half of the variables from each level
of multi-scale architecture. Following Kingma et al's implementation, the mean
and log-std. dev. are learned using a zero-initialized CNN.
'''
class Gaussianize(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        
        # The input channel number C should've already been halved
        self.C, self.H, self.W = input_shape
        self.z     = None
        self.const = -0.5 * math.log(2 * math.pi)
        self.cnn   = Conv2dZeroInit(self.C, 2 * self.C, 3, padding=1)

    def forward_flow(self, in_tensor, logdet):
        h, z    = torch.chunk(in_tensor, 2, dim=1)  # Factor out 1/2 of channels
        self.z  = z
        
        cnn_out = self.cnn(h)
        mean    = cnn_out[:, 0::2, :, :]
        logsd   = cnn_out[:, 1::2, :, :]

        log_pz  = self.const - logsd - 0.5 * ((z - mean) / torch.exp(logsd)) ** 2
        logdet  = torch.sum(log_pz, dim=(1, 2, 3)) + logdet

        if debug:
            cnn_out.register_hook(check_backprop('Gaussianize cnn'))

        return h, logdet

    def reverse_flow(self, in_tensor, mode):
        if (mode == 'generate'):
            # Generate new samples
            mean = torch.zeros(num_new_samples, self.C, self.H, self.W, dtype=torch.float, device=device)
            sd   = torch.ones(num_new_samples, self.C, self.H, self.W, dtype=torch.float, device=device)
            eps  = torch.randn_like(sd)
            z    = eps.mul(sd).add(mean)
        else:
            # The latent variables that are factored out before the final level
            # are stored during forward flow, and used for reconstruction in the
            # reverse flow.
            z       = self.z
            cnn_out = self.cnn(in_tensor)
            mean    = cnn_out[:, 0::2, :, :]
            logsd   = cnn_out[:, 1::2, :, :]
            
        return torch.cat((in_tensor, z), dim=1)
    
    def forward(self, in_tensor, logdet, direction):
        if (direction == 'forward'):
            return self.forward_flow(in_tensor, logdet)
        else:
            return self.reverse_flow(in_tensor, direction)

#---------------------
# Squeeze layer
#---------------------
'''
Operation that trade spatial size (H x W) for number of channels (C) [2]:
  Input (C x H x W) -> Output (s*C x H/s x W/s)
where s is the length of the subsquare.

Example: For s=2
Input:
  tensor([[[[ 1,  2,  5,  6,  9, 10],
            [ 3,  4,  7,  8, 11, 12]]]])
  torch.Size([1, 1, 2, 6])
  
Output:
  tensor([[[[ 1,  5,  9]],
           [[ 2,  6, 10]],
           [[ 3,  7, 11]],
           [[ 4,  8, 12]]]])
  torch.Size([1, 4, 1, 3])
'''
class Squeeze(nn.Module):
    def __init__(self, subsq = 2):
        super().__init__()
        
        self.subsq = subsq
        
    def squeeze(self, in_tensor):
        B, C_in, H_in, W_in = in_tensor.size()
        
        assert H_in % self.subsq == 0 and W_in % self.subsq == 0
        H_out = H_in // self.subsq
        W_out = W_in // self.subsq
        C_out = C_in * self.subsq * self.subsq
        
        out_tensor = in_tensor.view(B, C_in, H_out, self.subsq, W_out, self.subsq)
        out_tensor = out_tensor.permute(0, 1, 3, 5, 2, 4).contiguous()
        out_tensor = out_tensor.view(B, C_out, H_out, W_out)
            
        return out_tensor
    
    def unsqueeze(self, in_tensor):
        B, C_in, H_in, W_in = in_tensor.size()
        
        assert C_in >= 4 and C_in % 4 == 0
        H_out = H_in * self.subsq
        W_out = W_in * self.subsq
        C_out = C_in // (self.subsq * self.subsq)
        
        out_tensor = in_tensor.view(B, C_out, self.subsq, self.subsq, H_in, W_in)
        out_tensor = out_tensor.permute(0, 1, 4, 2, 5, 3).contiguous()
        out_tensor = out_tensor.view(B, C_out, H_out, W_out)
        
        return out_tensor
    
    def forward(self, in_tensor, logdet, direction):
        if (direction == 'forward'):
            return self.squeeze(in_tensor), logdet
        else:
            return self.unsqueeze(in_tensor)

#---------------------
# Flow step
#---------------------
'''
Activation normalization
Performs an affine transformation of the activations using a scale and bias
parameter per channel, similar to batch normalization. These parameters are
initialized such that the post-actnorm activations per-channel have zero mean
and unit variance given an initial minibatch of data. After initialization,
the scale and bias are treated as regular trainable parameters that are
independent of the data. [1]
'''
class ActNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        
        self.init_done = False
        self.bias  = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def reset_parameters(self, in_tensor):
        # Data dependent initialization
        # Mini-batch mean (bias) and variance (scale) per channel
        B, C, H, W   = in_tensor.size()
        num_elements = B * H * W
        mean         = torch.sum(in_tensor, dim=(0, 2, 3), keepdim=True) / num_elements
        var          = torch.sum((in_tensor - mean) ** 2, dim=(0, 2, 3), keepdim=True) / num_elements
        with torch.no_grad():
            self.bias.copy_(-mean)
            self.scale.copy_(1 / torch.sqrt(var + 1e-6)) # Add 1e-6 for numerical stability

        self.init_done = True
        return
                
    def forward_flow(self, in_tensor, logdet):
        if not self.init_done:
            self.reset_parameters(in_tensor)
        
        _, _, H, W = in_tensor.size()
        logscale   = torch.log(self.scale.squeeze())
        logdet     = torch.sum(logscale) * H * W + logdet
        out_tensor = (in_tensor + self.bias) * self.scale
        
        return out_tensor, logdet

    def reverse_flow(self, in_tensor):
        out_tensor = in_tensor / self.scale - self.bias
        return out_tensor

    def forward(self, in_tensor, logdet, direction):
        if (direction == 'forward'):
            return self.forward_flow(in_tensor, logdet)
        else:
            return self.reverse_flow(in_tensor)

class InvertibleConv(nn.Conv2d):
    def __init__(self, num_channels):
        self.C = num_channels
        super().__init__(in_channels=num_channels, out_channels=num_channels,
                         kernel_size=1, bias=False)
        
    # Override nn.Conv2d's reset function
    def reset_parameters(self):
        with torch.no_grad():
            # Initialise weights to a random rotation matrix
            w_init = np.linalg.qr(np.random.randn(self.C, self.C))[0]
            w_init = torch.from_numpy(w_init.astype('float32'))
            w_init = w_init.unsqueeze(-1).unsqueeze(-1)
            self.weight.copy_(w_init)
    
    def forward_flow(self, x, logdet):
        _, _, H, W = x.size()
        
        # Compute log-determinant of the weight matrix (wm). The log-determinant
        # of the Jacobian of 1x1 convolution is wm_logdet * H * W
        wm_logdet = torch.det(self.weight.squeeze()).abs().log()
        logdet    = wm_logdet * H * W + logdet
        
        y = F.conv2d(x, self.weight, self.bias, self.stride,
                     self.padding, self.dilation, self.groups)
        
        return y, logdet
        
    def reverse_flow(self, y):
        _, _, H, W = y.size()
                
        wm_inv = torch.inverse(self.weight.squeeze()).unsqueeze(-1).unsqueeze(-1)
        x      = F.conv2d(y, wm_inv, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        return x

    def forward(self, in_tensor, logdet, direction):
        if (direction == 'forward'):
            return self.forward_flow(in_tensor, logdet)
        else:
            return self.reverse_flow(in_tensor)
        
class AffineCoupling(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        
        self.C_in  = num_channels // 2
        self.C_out = num_channels
        self.C_mid = 512
        
        # The last convolution layer is initialized with zeros, so each affine
        # coupling layer initially performs an identity function which helps
        # with training very deep networks [1].
        self.cnn = nn.Sequential(
#                        Conv2dActNorm(self.C_in, self.C_mid, kernel_size=3, padding=1),
#                        nn.ReLU(inplace=True),
#                        Conv2dActNorm(self.C_mid, self.C_mid, kernel_size=1),
#                        nn.ReLU(inplace=True),
                        nn.Conv2d(self.C_in, self.C_mid, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(self.C_mid),
                        nn.Conv2d(self.C_mid, self.C_mid, kernel_size=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(self.C_mid),
                        Conv2dZeroInit(self.C_mid, self.C_out, kernel_size=3, padding=1),
#                        nn.Conv2d(self.C_mid, self.C_out, kernel_size=3, padding=1),
                   )
        
    def forward_flow(self, in_tensor, logdet):            
        x1, x2  = torch.chunk(in_tensor, 2, dim=1)
        cnn_out = self.cnn(x1)
        # REVISIT: exp() is said to be used for s in [1], but in their code
        #          sigmoid is actually used instead. No explanation was given
        #          but Nalisnick et al. think it might help condition the log
        #          likelihood in large models [3]. 
        #s      = torch.exp(cnn_out[:, 0::2, :, :] + 2.) # Scale
        s      = torch.sigmoid(cnn_out[:, 0::2, :, :] + 2.) # Log scale
        t      = cnn_out[:, 1::2, :, :]                     # Translation
        y1     = x1

        y2     = x2 * torch.exp(s) + t
        # Each s is a diagonal element of the Jacobian matrix
        logdet = torch.sum(torch.log(s), dim=(1, 2, 3)) + logdet
               
        if debug:
            cnn_out.register_hook(check_backprop('AffineCoupling cnn'))

        return torch.cat([y1, y2], dim=1), logdet
        
    def reverse_flow(self, in_tensor):
        y1, y2  = torch.chunk(in_tensor, 2, dim=1)
        cnn_out = self.cnn(y1)
        s       = torch.sigmoid(cnn_out[:, 0::2, :, :] + 2.) # Log scale
        t       = cnn_out[:, 1::2, :, :]                     # Translation
        x1      = y1
        x2      = (y2 - t) * torch.exp(-s)

        return torch.cat([x1, x2], dim=1)
      
    def forward(self, in_tensor, logdet, direction):
        if (direction == 'forward'):
            return self.forward_flow(in_tensor, logdet)
        else:
            return self.reverse_flow(in_tensor)

#---------------------
# Full model
#---------------------

class Glow(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        
        self.nn1_layers = nn.ModuleList()
        self.nn2_layers = nn.ModuleList()
        self.shared_layers = nn.ModuleList()
        
        # One level consists of 1 squeezing op and several flow steps
        C, H, W = input_shape
        for l in range(model_lvl):
            self.nn1_layers.append(Squeeze())
            C = C * 4
            H = H // 2
            W = W // 2
            print("Post squeeze C, H, W =", C, H, W)
            
            # One flow step consists of
            #  - Activation normalization layer
            #  - 1x1 convolution layer
            #  - Coupling layer
            for s in range(flow_steps):
                self.nn1_layers.append(ActNorm(C))
                self.nn1_layers.append(InvertibleConv(C))
                self.nn1_layers.append(AffineCoupling(C))
            
            # Factor out half of channels every level except the last
            if (l != model_lvl - 1):
                C = C // 2
                self.nn1_layers.append(Gaussianize((C, H, W)))
                print("Post factor-out C =", C)

        C, H, W = input_shape
        for l in range(model_lvl):
            self.nn2_layers.append(Squeeze())
            C = C * 4
            H = H // 2
            W = W // 2
            print("Post squeeze C, H, W =", C, H, W)
            
            # One flow step consists of
            #  - Activation normalization layer
            #  - 1x1 convolution layer
            #  - Coupling layer
            for s in range(flow_steps):
                self.nn2_layers.append(ActNorm(C))
                self.nn2_layers.append(InvertibleConv(C))
                self.nn2_layers.append(AffineCoupling(C))
            
            # Factor out half of channels every level except the last
            if (l != model_lvl - 1):
                C = C // 2
                self.nn2_layers.append(Gaussianize((C, H, W)))
                print("Post factor-out C =", C)
                
        # Shared latent variable
        self.shared_layers.append(ActNorm(C))
        self.shared_layers.append(InvertibleConv(C))
        self.shared_layers.append(AffineCoupling(C))
        self.shared_layers.append(GaussianPrior((C, H, W)))

    def encode(self, x1, x2, objective):
        h1, logdet1 = x1, objective
        h2, logdet2 = x2, objective

        for layer in self.nn1_layers:
            if debug:
                layer.register_forward_hook(printnorm)
                layer.register_backward_hook(printgrad)
                #if (layer.__class__.__name__ == 'FlowModule'):
                #    print(layer.one_step[2].cnn[6].weight)
                # For complex Module, use torch.Tensor.register_hook() directly
                # on a specific tensor to get the required gradients.

            h1, logdet1 = layer(h1, logdet1, 'forward')

        for layer in self.nn2_layers:
            h2, logdet2 = layer(h2, logdet2, 'forward')
            
        for layer in self.shared_layers:
            h1, logdet1 = layer(h1, logdet1, 'forward')
            h2, logdet2 = layer(h2, logdet2, 'forward')

        z1, objective1 = h1, logdet1
        z2, objective2 = h2, logdet2
        return z1, z2, objective1, objective2
    
    def decode1(self, z, mode):
        h = z
        for layer in reversed(self.nn1_layers):
            h = layer(h, None, mode)
        x = h
        return x
    
    def decode2(self, z, mode):
        h = z
        for layer in reversed(self.nn2_layers):
            h = layer(h, None, mode)
        x = h
        return x
    
    def forward(self, input_tensor1, input_tensor2, objective, direction):
        if (direction == 'forward'):
            return self.encode(input_tensor1, input_tensor2, objective)
        elif (direction == 'reverse_net1'):
            return self.decode1(input_tensor1, 'recon')
        else:
            return self.decode2(input_tensor2, 'recon')

    def gen_sample(self):
        return self.decode(None, 'generate')

###############################
# Main
###############################

model     = Glow((C, H, W)).to(device)
optimiser = optim.Adam(model.parameters(), lr=lr)
# Learning rate annealing
scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=45, gamma=0.1)

if not new_session:
    model.load_state_dict(saved_states['model'])
    optimiser.load_state_dict(saved_states['optimiser'])
    scheduler.load_state_dict(saved_states['scheduler'])

#print(model)
print("Number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))

# Approx. dequantization cost
dequan_cost = -np.log2(quan_lvl) * img_dim

def train(epoch):
    avg_bits_per_dim = RunningAverage() # Minibatch average
    start_time = time.time()
    model.train()
    for batch, (x1, x2, label) in enumerate(train_loader):
        with torch.autograd.detect_anomaly():
            optimiser.zero_grad()

            x1 = x1.to(device)
            x2 = x2.to(device)
            
            if (debug):
                # Funtion hooks will handle anomaly in debug mode
                torch.autograd.set_detect_anomaly(False)
                print("Epoch {}, batch {}".format(epoch, batch))
            
            # Note the images have been dequantized during dataloading
            
            # In case the number of training samples is not divisible by batch_size
            actual_batch_size = x1.size(0)

            # Log-likelihood
            # Approx. cost from dequantization of discrete data is a constant
            # and doesn't really matter in terms of training, but is added for
            # consistency so we can compare results against those in the literature.
            llh = dequan_cost + torch.zeros(actual_batch_size, device=device)

            z1, z2, llh1, llh2 = model(x1, x2, llh, 'forward')
            
            # Negative log-likelihood measuring expected compression cost in
            # bits per dimension.
            # Convert the total discrete log-likelihood to bits (log base 2) and
            # normalize by image dimension (e.g. 32x32x3 = 3072 for CIFAR-10).
            nll = -llh1 / (np.log(2.) * img_dim) - llh2 / (np.log(2.) * img_dim)
            nll = torch.mean(nll)
            
            # Cross domain reconstruction loss
            # The aim is to reconstruct x1 from z2, and x2 from z1
            recon1 = model(z2, None, None, 'reverse_net1') + 0.5 # +0.5 because we -0.5 during dequantization
            recon2 = model(None, z1, None, 'reverse_net2') + 0.5
            recon_loss = (F.binary_cross_entropy(recon1, x1, reduction='sum') +
                          F.binary_cross_entropy(recon2, x2, reduction='sum'))
            
            # REVISIT: Check the scale of nll v.s. recon_loss
            loss = nll + recon_loss
            loss.backward()

            nn.utils.clip_grad_value_(model.parameters(), 5)
            nn.utils.clip_grad_norm_(model.parameters(), 100)
            optimiser.step()

            avg_bits_per_dim.update(nll.item())

            if (debug):
                print("Input:")
                npimg = x.detach().to('cpu').numpy() + 0.5 # +0.5 because we -0.5 during dequantization
                show_images(np.clip(npimg[0:25], 0, 1), colour_img)
                print("Reconstruct:")
                recon = model(z, None, 'reverse')
                npimg = recon.detach().to('cpu').numpy() + 0.5
                show_images(np.clip(npimg[0:25], 0, 1), colour_img)
                
    train_objective.NLL.append(avg_bits_per_dim)
        
    epoch_time = time.time() - start_time
    print("Time Taken for Epoch {}: {:.2f}s".format(epoch, epoch_time))
    print('Epoch {} avg. training nll: {:.3f}'.format(epoch, nll.item()))
    
    if epoch % 1 == 0:
        with torch.no_grad():
            print("Epoch {} training cycle-consistency reconstruction 1:".format(epoch))
            recon = model(z2, None, 'reverse')
            npimg = recon.detach().to('cpu').numpy() + 0.5 # +0.5 because we -0.5 during dequantization
            show_images(np.clip(npimg[0:25], 0, 1), colour_img)
            print("Epoch {} training cycle-consistency reconstruction 2:".format(epoch))
            recon = model(z1, None, 'reverse')
            npimg = recon.detach().to('cpu').numpy() + 0.5 # +0.5 because we -0.5 during dequantization
            show_images(np.clip(npimg[0:25], 0, 1), colour_img)
            #print("Epoch {} new sample:".format(epoch))
            #sample = model.gen_sample()
            #npimg  = sample.detach().to('cpu').numpy() + 0.5
            #show_images(np.clip(npimg[0:25], 0, 1), colour_img)
    return

def validation(epoch):
    avg_bits_per_dim = RunningAverage()
    model.eval()
    with torch.no_grad():
        for batch, (x1, x2, label) in enumerate(val_loader1):
            x1 = x1.to(device)
            x2 = x2.to(device)
            actual_batch_size  = x1.size(0)
            llh = dequan_cost + torch.zeros(actual_batch_size, device=device)
            z1, z2, llh1, llh2 = model(x1, x2, llh, 'forward')
            nll = -llh1 / (np.log(2.) * img_dim) - llh2 / (np.log(2.) * img_dim)
            nll = torch.mean(nll)
            recon1 = model(z2, None, None, 'reverse_net1') + 0.5 # +0.5 because we -0.5 during dequantization
            recon2 = model(None, z1, None, 'reverse_net2') + 0.5
            recon_loss = (F.binary_cross_entropy(recon1, x1, reduction='sum') +
                          F.binary_cross_entropy(recon2, x2, reduction='sum'))
            # REVISIT: Check the scale of nll v.s. recon_loss
            loss = nll + recon_loss
            
            avg_bits_per_dim.update(loss.item())
            
        val_objective.NLL.append(avg_bits_per_dim)

        if epoch % 10 == 0:
            print("Epoch {} validation reconstruction:".format(epoch))
            recon = model(z, objective, 'reverse')
            npimg = recon.detach().to('cpu').numpy() + 0.5 # +0.5 because we -0.5 during dequantization
            show_images(np.clip(npimg[0:25], 0, 1), colour_img)

    return

for epoch in range(start_epoch, num_epochs + 1):
    scheduler.step()
    train(epoch)
    validation(epoch)

    # Plot losses
    if in_colab:
        plot_loss(grid, 1, epoch, train_objective, val_objective)

    if epoch % 5 == 0:
        # Save checkpoints
        torch.save({
            'epoch'           : epoch + 1,
            'model'           : model.state_dict(),
            'optimiser'       : optimiser.state_dict(),
            'scheduler'       : scheduler.state_dict(),
            'train_objective' : train_objective,
            'val_objective'   : val_objective,
            'hyperparams'     : {'batch_size' : batch_size,
                                 'num_epochs' : num_epochs,
                                 'lr'         : lr,
                                 'model_lvl'  : model_lvl,
                                 'flow_steps' : flow_steps}
        }, '{}/epoch_{}.pth'.format(chkpt_dir, epoch))


torch.save({
    'epoch'           : epoch + 1,
    'model'           : model.state_dict(),
    'optimiser'       : optimiser.state_dict(),
    'scheduler'       : scheduler.state_dict(),
    'train_objective' : train_objective,
    'val_objective'   : val_objective,
    'hyperparams'     : {'batch_size' : batch_size,
                         'num_epochs' : num_epochs,
                         'lr'         : lr,
                         'model_lvl'  : model_lvl,
                         'flow_steps' : flow_steps}
}, '{}/epoch_{}.pth'.format(chkpt_dir, epoch))
