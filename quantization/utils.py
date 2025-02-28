import torch
import numpy as np
import os
import pickle

def list_has_common_element(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return len(set1.intersection(set2)) > 0

def format_string_with_condition(input_string, condition, symm, fbit, bbit, row_block, column_block, input_pad=50, ):
    padded_string = input_string.ljust(input_pad)
    if condition:
        output_string = padded_string + "True".ljust(10) + "".ljust(10)
    else:
        output_string = padded_string + "".ljust(10) + "False".ljust(10)

    output_string = output_string + f"Symm {symm}".ljust(20) + \
                    f"Forward bit {fbit}".ljust(20) + f"Backward bit {bbit}".ljust(20) + \
                    f"Row Blocksize {row_block}".ljust(20) + f"Column Blocksize {column_block}".ljust(20)
    return output_string

def print_warning(sentence):
    print("*" * (len(sentence) + 4))
    print(f"* {sentence} *")
    print("*" * (len(sentence) + 4))

def check_nan_inf(tensor, check_nan, check_inf):
    if check_nan:
        contain_nan = torch.isnan(tensor).any()
    else:
        contain_nan = False
    if check_inf:
        contain_inf = torch.isinf(tensor).any()
    else:
        contain_inf = False
    return contain_nan, contain_inf

def move_torch_to_numpy(tensor):
    if tensor is None:
        return None

    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.detach().float().numpy()

def flatten_to_1d(tensor):
    if tensor is None:
        return None

    return tensor.reshape(-1)
