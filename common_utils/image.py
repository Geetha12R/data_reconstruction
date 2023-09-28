__author__ = "Niv Haim, Weizmann Institute of Science"

import torch
import kornia.metrics as metrics
from tqdm.auto import tqdm
# from kornia.losses.multiscale_ssim import ms_ssim
from PIL import Image
import torch.nn.functional as F

def resize_tensors_to_min_size(tensors, min_size=32):
    resized_tensors = []
    
    for tensor in tensors:
        # Get the current height and width
        h, w = tensor.shape[-2], tensor.shape[-1]
        
        if h < min_size or w < min_size:
            # Resize the tensor to meet the minimum size requirement
            scale_factor = max(min_size / h, min_size / w)
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            
            tensor = F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        resized_tensors.append(tensor)
    
    return resized_tensors


def get_ssim_pairs_kornia(x, y, type='dssim'):

    x_resized, y_resized = resize_tensors_to_min_size([x, y])
 
    if type == 'mssim':
        return ms_ssim(x_resized, y_resized, win_size=3).reshape(x_resized.shape[0], -1).mean(dim=1)
    if type == 'dssim':
        return metrics.ssim(x, y, window_size=3).reshape(x.shape[0], -1).mean(dim=1)


def get_ssim_all(x, y, metric):
    ssims = []
    for i in tqdm(range(y.shape[0])):
        scores = get_ssim_pairs_kornia(x, y[i:i + 1].expand(x.shape[0], -1, -1, -1), metric)
        ssims.append(scores)

    return torch.stack(ssims).t()

# Example usage:
# x and y should be torch tensors with shape (batch_size, channels, height, width)
# ms_ssim_matrix = get_ms_ssim_all(x, y)



# import numpy as np
# import skimage.io
# import os
# from skimage.metrics import structural_similarity
# from skimage.color import rgb2gray


# #       The Base!
# #####################
# def img_read(img_path, device, m11=False):
#     x = skimage.io.imread(img_path)
#     if len(x.shape) == 2:
#         x = np.expand_dims(x, -1)
#     gt = np.expand_dims(x, 0)
#     gt = torch.tensor(gt).contiguous().permute(0,3,1,2).detach().to(device)
#
#     if m11:
#         return img_255_to_m11(gt)
#     else:
#         return gt
#
#
# #####################
# #       Base coversions etc.
# #####################
# def img_255_to_m11(x):
#     return x.div(255).mul(2).add(-1)
#
#
# def img_m11_to_255(x):
#     return x.add(1).div(2).mul(255)
#
#
# #####################
# #####################
#
# def tensor2npimg(x, vmin=-1, vmax=1, normmaxmin=False):
#     """tensor in [-1,1] (1x3xHxW) --> numpy image ready to plt.imshow"""
#     if normmaxmin:
#         vmin = x.min().item()
#         vmax = x.max().item()
#     final = x[0].add(-vmin).div(vmax-vmin).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0)
#     # if input has 1-channel, pass grayscale to numpy
#     if final.shape[-1] == 1:
#         final = final[:,:,0]
#     return final.to('cpu', torch.uint8).numpy()
#
#
# torch255tonpimg = lambda x: x[0].add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
#
# #####################
#
# def get_2d_grid(h, w):
#     eye2d = torch.tensor([[[1, 0, 0], [0, 1, 0]]]).float()
#     grid = torch.nn.functional.affine_grid(theta=eye2d, size=(1, 1, h, w), align_corners=False)
#     grid = grid.permute(0, 3, 1, 2)
#     return grid
#
#
# def get_3d_grid(t, h, w):
#     eye3d = torch.tensor([[[1, 0, 0], [0, 1, 0]]]).float()
#     grid = torch.nn.functional.affine_grid(theta=eye3d, size=(1, 1, t, h, w), align_corners=False)
#     grid = grid.permute(0, 4, 1, 2, 3)
#     return grid
#
#
# #####################
# #       Image Metrics
# #####################
#
# def psnr(im, ref, margin=2):
#     # assume images are tensors float 0-1.
#     rgb2gray = torch.Tensor([65.481, 128.553, 24.966]).to(im.device)[None, :, None, None]
#     gray_im, gray_ref = torch.sum(im * rgb2gray, dim=1) + 16, torch.sum(ref * rgb2gray, dim=1) + 16
#     clipped_im, clipped_ref = gray_im.clamp(0, 255).squeeze(), gray_ref.clamp(0, 255).squeeze()
#     shaved_im, shaved_ref = clipped_im[margin:-margin, margin:-margin], clipped_ref[margin:-margin, margin:-margin]
#     return 20 * torch.log10(torch.tensor(255.)) -10.0 * ((shaved_im) - (shaved_ref)).pow(2.0).mean().log10()
#
#
# def get_means_stds(x):
#     n, c, h, w = x.shape
#     means = x.reshape(n * c, h * w).mean(dim=1).reshape(n, c, 1, 1)
#     stds = x.reshape(n * c, h * w).std(dim=1).reshape(n, c, 1, 1)
#     return means, stds
#
#
# def get_ssim_pairs_skimage(x, y):
#     def ssim(true, pred):
#         pred = rgb2gray(pred)
#         true = rgb2gray(true)
#         L = pred.max() - pred.min()
#         return structural_similarity(pred, true, data_range=L, K1=0.01, K2=0.03)
#
#     ssims = []
#     for i in range(x.shape[0]):
#         score = ssim(x[i].permute(1, 2, 0).cpu().numpy(), y[i].permute(1, 2, 0).cpu().numpy())
#         ssims.append(score)
#     ssims = torch.tensor(ssims)
#     return ssims


