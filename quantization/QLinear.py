import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

from .Qconfig import qconfig
from .utils import *
from .QFunction import *

import os
from copy import deepcopy

from torch.cuda import amp

class QLinearLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, args=None, layer_type=''):
        super(QLinearLayer, self).__init__(in_features, out_features, bias)
        self.args = args
        self.layer_type = layer_type
        assert layer_type != '', "layer_type is not defined"
        assert layer_type in qconfig.qlinear_config.keys(), f"{layer_type} not in qlinear_config"

        self.apply_quantize = list_has_common_element(args.qchoice, qconfig.qlinear_config[layer_type])

        self.fbit = self.args.fwbit if self.args.fwbit else self.Ubit
        self.bbit = self.args.bwbit if self.args.bwbit else self.Ubit
        quantize_flag = format_string_with_condition(layer_type, self.apply_quantize, self.args.symm, self.fbit, self.bbit,
                                                     self.args.row_blocksize, self.args.column_blocksize)
        print(quantize_flag)
        self.frozen_args = deepcopy(self.args)
        
        if self.apply_quantize and args.qlinear_ema_decay > 1e-10:
            self.is_ema = True
            self.ema_decay = args.qlinear_ema_decay
            
            self.register_buffer('ema_step', torch.tensor(0, dtype=torch.int))
            self.register_buffer('ema_weight', self.weight.data.clone())
        else:
            self.is_ema = False

    def forward(self, input):
        
        """
        We update W_EMA During Training Process (in engine.py/train_one_epoch)
        if args.qlinear_ema_decay > 0:         # EMA-Weight Update
            for name, module in model.module.named_modules():
                if isinstance(module, QLinearLayer):
                    if module.training and module.apply_quantize and module.is_ema:
                        module.ema_step += 1
                        module.ema_weight.mul_(module.ema_decay) \
                                         .add_(module.weight.data, alpha = 1 - module.ema_decay)
        """
        
        
        if self.apply_quantize and self.is_ema and self.ema_step >= 10 and self.args.qlinear_ema_decay > 1e-10:
            output = QuantLinear.apply(input, 
                                    self.weight, self.bias, self.ema_weight,
                                    self.args, self.layer_name, self.apply_quantize)
        else:
            output = QuantLinear.apply(input, 
                                    self.weight, self.bias, None,
                                    self.args, self.layer_name, self.apply_quantize)

        return output

class QuantLinear(Function):
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, input, weight, bias, ema_weight,
                args, layer_name, apply_quantize=True):
        
        ######## [Q1] Qinput, Iscale = Q_D(X)  1x32 ##########
        Binput = block_cut(input, args.row_blocksize, args.column_blocksize)
        RQinput, Qinput, Iscale = block_quant(
            Binput, args.symm, args.fabit, exp_bits=args.faexp,
            stochastic=False, epsilon=args.epsilon, 
            apply_quantize=apply_quantize and (args.qlinear_f_a_in or args.qlinear_all), 
            use_triton=args.tritonQ,
            MXScale=args.mxscale)
        
        Qinput  = block_reshape( Qinput, input, args.row_blocksize, args.column_blocksize)
        RQinput = block_reshape(RQinput, input, args.row_blocksize, args.column_blocksize)

        ######## [Q2] Qweight, Wscale = Q_D(W)  1x32 ##########
        ### NOTE: We leave transpose to F.linear, so the blocksize here is 1x32
        
        Bweight = block_cut(weight, args.row_blocksize, args.column_blocksize)
        
        if not (ema_weight is None):
            Bema_weight = block_cut(ema_weight, args.row_blocksize, args.column_blocksize)
        else:
            Bema_weight = None
        
        RQweight, Qweight, Wscale = block_quant(
            Bweight, args.symm, args.fwbit, exp_bits=args.fwexp,
            stochastic=False, epsilon=args.epsilon, 
            apply_quantize=apply_quantize and (args.qlinear_f_w_in or args.qlinear_all),
            ema_input=Bema_weight, use_triton=args.tritonQ,
            MXScale=args.mxscale)
        
        Qweight = block_reshape(Qweight, weight, args.row_blocksize, args.column_blocksize)
        RQweight = block_reshape(RQweight, weight, args.row_blocksize, args.column_blocksize)

        # save Q_D(X), Q_D(W) for backward
        ctx.saved = Qinput, Iscale, Qweight, Wscale, bias, args, layer_name, apply_quantize
        
        # Linear output
        fc_output = F.linear(RQinput, RQweight, bias) 
        
        return fc_output

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        Qinput, Iscale, Qweight, Wscale, bias, args, layer_name, apply_quantize = ctx.saved

        Bgrad_output   = block_cut(grad_output, args.row_blocksize,    args.column_blocksize)  # RB x CB = 1 x 32
        Bgrad_output_t = block_cut(grad_output, args.column_blocksize, args.row_blocksize)     # CB x RB = 32 x 1
        
        ######## [Q3] Q_S ( ∇_Y )  1x32 ##########
        RQgrad_output, Qgrad_output, _ = block_quant(
            Bgrad_output, args.symm, args.babit, exp_bits=args.baexp,
            stochastic=True, epsilon=args.epsilon, 
            apply_quantize=apply_quantize and (args.qlinear_b_dy or args.qlinear_all),
            use_triton=args.tritonQ,
            MXScale=args.mxscale)
        
        ######## [Q5] Q_S ( ∇_Y )  32x1 ##########
        ### NOTE: We leave transpose outside quantization, so the blocksize here is 1x32
        RQgrad_output_t, Qgrad_output_t, _ = block_quant(
            Bgrad_output_t, args.symm, args.babit, exp_bits=args.bwexp,
            stochastic=True, epsilon=args.epsilon, 
            apply_quantize=apply_quantize and (args.qlinear_b_dy_t or args.qlinear_all),
            use_triton=args.tritonQ,
            MXScale=args.mxscale)
        
        # get quantize-dequantize   ∇_Y(1x32) & ∇_Y(32x1)
        grad_output   = block_reshape(RQgrad_output,   grad_output, args.row_blocksize,    args.column_blocksize)
        grad_output_t = block_reshape(RQgrad_output_t, grad_output, args.column_blocksize, args.row_blocksize)

        #=======================∇_W Calculation=====================
        ######## [Q6]  Q_S ( Dequantize[Q_D(X)] 1x32 )  32x1 ########
        # [Q6]-[1]      Dequantize[Q_D(X)] 1x32
        Binput = block_cut(Qinput, args.row_blocksize, args.column_blocksize)
        Dinput = Binput * Iscale
        Dinput = block_reshape(Dinput, Qinput, args.row_blocksize, args.column_blocksize)
        
        # [Q6]-[2]      Q_S ( Dequantize[Q_D(X)] 1x32 )  32x1
        BRinput = block_cut(Dinput, args.column_blocksize, args.row_blocksize)            
        QRinput, _, _ = block_quant( # [x] TODO: determine `dtype` here 
            BRinput, args.symm, args.fabit, exp_bits=args.faexp, 
            stochastic=True, epsilon=args.epsilon, 
            apply_quantize=apply_quantize and (args.qlinear_b_a_rq or args.qlinear_all),
            use_triton=args.tritonQ,
            MXScale=args.mxscale)
        
        Rinput = block_reshape(QRinput, Qinput, args.column_blocksize, args.row_blocksize)

        grad_output_flatten_t = grad_output_t   .reshape(-1, grad_output_t.shape[-1])
        input_flatten         = Rinput          .reshape(-1, Qinput.shape[-1])

        #   ∇_W Calculation
        grad_weight = grad_output_flatten_t.t().mm(input_flatten)
        
        #=======================∇_X Calculation=====================
        ######## [Q4] Q_S ( Dequantize[Q_D(W)] 1x32 )  32x1 ########
        # [Q4]-[1]      Dequantize[Q_D(W)] 1x32
        Bweight = block_cut(Qweight, args.row_blocksize, args.column_blocksize)
        Dweight = Bweight * Wscale
        Dweight = block_reshape(Dweight, Qweight, args.row_blocksize, args.column_blocksize) 
        
        # [Q4]-[2]      Q_S ( Dequantize[Q_D(W)] 1x32 )  32x1
        BRweight = block_cut(Dweight, args.column_blocksize, args.row_blocksize)            
        QRweight, _, _ = block_quant( # [x] TODO: determine `dtype` here 
            BRweight, args.symm, args.fwbit, exp_bits=args.fwexp, 
            stochastic=True, epsilon=args.epsilon, 
            apply_quantize=apply_quantize and (args.qlinear_b_w_rq or args.qlinear_all),
            use_triton=args.tritonQ,
            MXScale=args.mxscale)
        
        Rweight = block_reshape(QRweight, Qweight, args.column_blocksize, args.row_blocksize)

        # ∇_X Calculation
        grad_output_flatten = grad_output.reshape(-1, grad_output_t.shape[-1])
        grad_input = grad_output_flatten.mm(Rweight)
        grad_input_transform = grad_input.reshape(Qinput.size())
        
        # ∇_Bias Calculation
        if bias is not None:
            grad_bias = grad_output_flatten.sum(0)
        else:
            grad_bias = None

        return grad_input_transform, grad_weight, grad_bias, None, None, None, None