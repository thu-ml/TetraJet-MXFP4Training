import torch
import math

try:
    from .QuantizerFP_torch import *
    from .QuantizerFP_triton import *
except ImportError:
    from QuantizerFP_torch import *
    from QuantizerFP_triton import *

def block_cut(input, row_block, column_block):
    # print(input.shape)
    original_shape = input.shape
    # input tensor shape is M * N
    if len(input.shape) > 2:
        input = input.reshape(-1, input.shape[2])
    elif len(input.shape) == 2:
        pass
    else:
        raise ValueError(f"input shape {input.shape} does not match for block cut, {input}")
    M, N = input.shape[0], input.shape[1]

    if row_block == -1:
        row_block = M
    if column_block == -1:
        column_block = N

    row_num, column_num = M // row_block, N // column_block

    assert row_num * row_block == M, f"{row_num}, {row_block}, {M}, {original_shape}"
    assert column_num * column_block == N, f"{column_num}, {column_block}, {N}, {original_shape}"
    # print(input.shape)
    input = input.reshape(row_num, row_block, column_num, column_block).permute(0, 2, 1, 3) \
        .reshape(row_num * column_num, row_block, column_block)
    # print(input.shape)
    return input

def block_reshape(input, origin_input, row_block, column_block):
    if len(origin_input.shape) > 2:
        flatten_input = origin_input.reshape(-1, origin_input.shape[2])
    elif len(origin_input.shape) == 2:
        flatten_input = origin_input
    else:
        raise ValueError(f"input shape {input.shape} does not match for block cut, {input}")

    M, N = flatten_input.shape[0], flatten_input.shape[1]

    if row_block == -1:
        row_block = M
    if column_block == -1:
        column_block = N

    row_num, column_num = M // row_block, N // column_block

    input = input.reshape(row_num, column_num, row_block, column_block).permute(0, 2, 1, 3) \
        .reshape(row_num * row_block, column_num * column_block)

    if len(origin_input.shape) > 2:
        input = input.reshape(origin_input.shape)
    elif len(origin_input.shape) == 2:
       pass
    else:
        raise ValueError(f"input shape {input.shape} does not match for block cut, {input}")

    return input

def block_quant(input, symm, 
                bits, exp_bits, 
                stochastic, epsilon, apply_quantize,
                ema_input = None, use_triton = False, 
                MXScale = 0):
    if symm:
        if ema_input is None:
            if use_triton:
                return SymmQuantizer_Triton.apply(input, bits, exp_bits, stochastic, epsilon, apply_quantize, MXScale)
            else:
                return SymmQuantizer_Torch .apply(input, bits, exp_bits, stochastic, epsilon, apply_quantize, MXScale)
        else:
            # NOTE: using SymmQuantizer_with_EMA means — stochastic == False 
            if use_triton:
                return SymmQuantizer_with_EMA_Triton.apply(input, bits, exp_bits, epsilon, apply_quantize, ema_input, MXScale)
            else:
                return SymmQuantizer_with_EMA_Torch .apply(input, bits, exp_bits, epsilon, apply_quantize, ema_input, MXScale)
    else:
        raise NotImplementedError("Asym not implemented here")
