import torch
from torch.autograd.function import InplaceFunction, Function
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import time

# from einops import rearrange, repeat, reduce

''' Q_core.py
    Implemented some core modules & methods which are required for quantization, including:

    1. Block Floating Point (BFP) Basis
        -   BFP2FP & FP2BFP
            -   FP Decomposer
        -   BFP Operator
            -   Reshape
            -   Max, Mean, Var, ... along specific dimensions

    2. Quantization Module
        -   Stochastic Round
        -   BFP/INT/FP Quantizer
    
    3. Online BFP Adptive Quantization(OBAQ) Module
        -   Sensitivity Analysis
        -   Bwmap Generator

    4. (Scalable) Custom Quantization Module

'''

### BFP Basis

def BFP_padding(data, padding_shape):  # 如果形状不符合，对
    device = data.device
    padding_size = list(np.array(padding_shape) - np.array(data.shape))
    if padding_size[1]:
        data = torch.cat([data, torch.zeros(data.shape[0], padding_size[1], data.shape[2], data.shape[3], device=device)], dim=1)
    if padding_size[0]:
        data = torch.cat([data, torch.zeros(padding_size[0], data.shape[1], data.shape[2], data.shape[3], device=device)], dim=0)
    return data

def get_BFP_shape(data_shape, block_size):
    # print('get_BFP_shape:', data_shape, block_size)
    return list(np.ceil(np.array(data_shape) / block_size).astype(np.int))

def get_BFP_paddingshape(data_shape, block_size):
    BFPshape = get_BFP_shape(data_shape, block_size)
    BFP_paddingshape = list(np.array(BFPshape) * np.array(block_size)) 
    return BFP_paddingshape

def BFP_block(data, block_size):    # data -> data_block
    BFPshape = get_BFP_shape(data.shape, block_size)
    data_block = data.reshape(
                            BFPshape[0], block_size[0],     
                            BFPshape[1], block_size[1],
                            BFPshape[2], block_size[2],
                            BFPshape[3], block_size[3]).permute(0, 2, 4, 6, 1, 3, 5, 7)
    return data_block
    # return rearrange(data, '(oc_b bs0) (ic_b bs1) (kx_b bs2) (ky_b bs3) -> oc_b ic_b kx_b ky_b bs0 bs1 bs2 bs3', 
    #                  bs0=block_size[0], bs1=block_size[1], bs2=block_size[2], bs3=block_size[3])

def BFP_deblock(data_block, shape):
    data = data_block.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(shape)
    return data
    # return rearrange(data_block, 'd0 d1 d2 d3 bs0 bs1 bs2 bs3 -> (d0 bs0) (d1 bs1) (d2 bs2) (d3 bs3)')

def BFP_max(data, block_size):
    data_block = BFP_block(data, block_size)
    s = data_block.shape
    data_block_max = data_block.reshape(s[0], s[1], s[2], s[3], -1).max(axis=4).values
    return data_block_max

def BFP_absmax(data, block_size):
    return BFP_max(data=data.abs(), block_size=block_size)

def BFP_norm(data, block_size):
    data_block = BFP_block(data, block_size)
    s = data_block.shape
    data_block_norm = data_block.reshape(s[0], s[1], s[2], s[3], -1).norm(dim=4)
    return data_block_norm

### Quantization Module

def round(data, scale, stochastic=True, min_value = None, max_value=torch.inf):
    if stochastic:
        noise = data.new(data.shape).uniform_(-0.5, 0.5)
    else:
        noise = 0
    if not isinstance(max_value, float):
        max_value[max_value<0]=0
    else:
        max_value = max(max_value, 0)
    data_quantized = (data / scale).add_(noise).round_().clip_(min_value, max_value).mul_(scale) # 非原位操作, 需要python3.7以上版本
    return data_quantized

def decompose_tensor(x:torch.Tensor):
    if x.dtype == torch.float32:
        mantissa_width = 23
        exponent_offset = 127
        view_type = torch.int32
    elif x.dtype == torch.float16:
        mantissa_width = 10
        exponent_offset = 15
        view_type = torch.int16
    elif x.dtype == torch.bfloat16:
        mantissa_width = 7
        exponent_offset = 127
        view_type = torch.int16
    else:
        raise TypeError(f"Unsupported data type: {x.dtype}")

    negative = x < 0
    n = torch.abs(x).view(view_type)
    exponent = (n >> mantissa_width) - exponent_offset
    mantissa = n & torch.tensor(((1 << mantissa_width) - 1), dtype=view_type)
    return negative, exponent.float(), mantissa

def BFPQuant(data:torch.tensor, block_size:tuple, block_bw:tuple, stochastic=False, sparsity_counter=None):

    if block_size is None or block_bw is None:  # block_size或block_bw为None时，不进行量化
        return data 
    
    with torch.no_grad():
        BFPshape = list(block_bw.shape)
        data_shape = data.shape
        data_padding_shape = list(np.array(BFPshape) * np.array(block_size)) 
        data_padding = BFP_padding(data, data_padding_shape)
        data_block = BFP_block(data_padding, block_size)                   # data block with padding
        _, exponent_block, _ = decompose_tensor(data_block)
        # _, exponent, _ = decompose_tensor(data_padding)
        exponent_max = exponent_block.reshape(BFPshape[0], BFPshape[1], BFPshape[2], BFPshape[3], -1).max(axis=4).values # get max exponent
        # exponent_max = BFP_max(exponent, block_size)
        if sparsity_counter is not None: 
            non_zeros = (exponent_max > -31) & (block_bw != 0)
            # if data.shape[0] == 128:
            #     print(torch.count_nonzero(block_bw == 0))
            #     print(block_bw[:4, :4, 0,0].cpu().numpy())
            sparsity = 1 - torch.count_nonzero(non_zeros) / np.product(BFPshape)
            sparsity_counter.update(sparsity.cpu())

        bins = (torch.tensor(1) << (block_bw-1))
        delta_block = (torch.tensor(1) << (exponent_max + 1)) / bins
        
        delta_block = torch.tile(delta_block[:, :, :, :, None, None, None, None], block_size)
        bins = torch.tile(bins[:,:,:,:,None, None, None,None], block_size)
        
        data_block = round(data_block, delta_block, stochastic, -bins+1, bins-1)

        data_quantized = BFP_deblock(data_block, data_padding_shape)                                  # data deblock
        data_quantized = data_quantized[:data_shape[0], :data_shape[1], :data_shape[2], :data_shape[3]] # clip shape
        return data_quantized

def INTQuant(data:torch.Tensor, bw, stochastic=False, mode='absmax'):
    if mode == 'exp':
        _, exponent, _ = decompose_tensor(data)
        exponent_max = exponent.max()
        mx = (torch.tensor(1) << (exponent_max+1)) + 1e-10
    elif mode == 'absmax':
        mx = data.abs().max() + 1e-10
    elif mode == 'mxmn':
        mx = data.max() + 1e-10
        mn = data.min() - 1e-10
    if mode == 'mxmn':
        delta = (mx - mn) / ((torch.tensor(1) << bw) - 1)
        data_quantized = round(data-mn, delta, stochastic) + mn
    else:
        delta = mx / ((torch.tensor(1) << (bw - 1)))
        data_quantized = round(data, delta, stochastic)
    # if mode == 'exp' and mx > 32:
    #     print(mx)
    return data_quantized
    
def FPQuant(data:torch.tensor, stochastic=False): ## S:1bit, E:4bit, M:3bit
    _, exponent, _ = decompose_tensor(data)
    exponent_max = exponent.max()
    print(exponent_max)
    mx = torch.tensor(1) << (torch.max(exponent, exponent_max-15) + 1) # 4 bit Exponent
    scale = mx / 8 # 1 bit Sign + 3 bit Mantissa 
    print(torch.round(data/scale))
    data_quantized = round(data, scale, stochastic)
    return data_quantized

### OBAQ-Module
def Sensitivity_Analysis(data, grad, block_size, C):
    data = data.detach().clone()
    grad = grad.detach().clone()
    BFP_paddingshape = get_BFP_paddingshape(data.shape, block_size)
    data_padding = BFP_padding(data, BFP_paddingshape)
    grad_padding = BFP_padding(grad, BFP_paddingshape)
    data_absmax = BFP_absmax(data_padding, block_size=block_size)
    grad_norm = BFP_norm(grad_padding, block_size=block_size)

    sensitivity = torch.log2(data_absmax * grad_norm / C + 1e-12)
    return sensitivity

def mean_bwmap(q_params_list, datatype, bwmaptype):
    layer_mean_bws = []
    layer_weights = []
    for q_params in q_params_list:
        bwmap = getattr(q_params, bwmaptype)
        layer_mean_bws.append(np.average(bwmap[datatype].cpu().numpy()))
        layer_weights.append(q_params.computations[datatype])
    
    # print(layer_mean_bws, layer_weights)
    total_mean_bw = np.average(layer_mean_bws, weights=layer_weights)
    return total_mean_bw, layer_mean_bws

def mean_sparsity(q_params_list, datatype):
    layer_mean_sparsities = []
    layer_weights = []
    for q_params in q_params_list:
        layer_mean_sparsities.append(q_params.sparsity_counter[datatype].avg)
        layer_weights.append(q_params.computations[datatype])
    print(datatype, layer_mean_sparsities, layer_weights)
    total_mean_sparsity = np.average(layer_mean_sparsities, weights=layer_weights)
    return total_mean_sparsity, layer_mean_sparsities