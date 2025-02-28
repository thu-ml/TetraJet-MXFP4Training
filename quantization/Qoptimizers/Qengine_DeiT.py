import math
import sys
from typing import Iterable, Optional

import torch
import torch.distributed as dist

from timm.data import Mixup
from timm.utils import accuracy

from losses import DistillationLoss
import utils
import dllogger as DLLogger

from copy import deepcopy

from quantization import QLinearLayer
from quantization import block_cut, block_quant, block_reshape

import os

def save_model(model, optimizer, loss_scaler, args, save_path):
    model_without_ddp = model.module if args.distributed else model
    
    parent_dir = os.path.dirname(save_path)
    os.makedirs(parent_dir, exist_ok=True)
    
    if dist.get_rank() == 0:    # Only the main process saves the model
                                # Save model, optimizer, and scaler states directly to file
        torch.save({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': loss_scaler.state_dict(),
        }, save_path)
    
    dist.barrier()
    
def resume_model(model, optimizer, loss_scaler, args, device, load_path):
    model_without_ddp = model.module if args.distributed else model
    
    
    checkpoint = torch.load(load_path, map_location=device)

    model_without_ddp.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss_scaler.load_state_dict(checkpoint['scaler'])
    
    dist.barrier()  # Ensure all processes are synchronized

def BSRamping_calib_update_period(model: torch.nn.Module,
                      criterion: DistillationLoss,
                      optimizer: torch.optim.Optimizer,
                      calib_data_loader: Iterable,
                      device: torch.device,
                      loss_scaler, max_norm: float = 0,
                      max_step: int = 1,
                      mixup_fn: Optional[Mixup] = None,
                      args = None,):

    map_to_update_period = {}
    save_model(model, optimizer, loss_scaler, args, args.opt_savepath)
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    optimizer.clear_for_ramping() # set optimizer to normal AdamW
    calib_steps = 30
    last_weights = {}
    last_RQweights = {}
    dis_weight = {}
    dis_RQweight = {}
            
    for step, (samples, targets) in enumerate(calib_data_loader):
        if step == calib_steps + 1:
            break
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if args.cosub:
            raise NotImplementedError
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)
        
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step #{step}")
            sys.exit(1)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        for name, module in model.module.named_modules():
            if isinstance(module, QLinearLayer):
                this_weight = module.weight.data.clone()
                Bweight = block_cut(this_weight, args.row_blocksize, args.column_blocksize)
                RQweight, _, _ = block_quant(
                    Bweight, args.symm, args.fwbit, exp_bits=args.fwexp,
                    stochastic=False, epsilon=args.epsilon, 
                    apply_quantize=True,
                    ema_input=None,
                    MXScale=args.mxscale)  # [Q2]
                RQweight = block_reshape(RQweight, this_weight, args.row_blocksize, args.column_blocksize)
                
                if step >= 1:
                    if name not in dis_weight:
                        dis_weight[name] = torch.zeros_like(this_weight)
                        dis_RQweight[name] = torch.zeros_like(this_weight)
                    
                    dis_weight[name]   += (this_weight - last_weights[name]).abs()
                    dis_RQweight[name] += (RQweight - last_RQweights[name]).abs()
                
                EPS = 1e-8
                if step == calib_steps:
                    map_to_update_period[module.weight] = \
                        torch.div(
                                    dis_RQweight[name] / args.opt_ramping_alpha, 
                                    dis_weight[name] + EPS, 
                                    rounding_mode='floor'
                                ) * args.opt_ramping_beta + 1
                    
                    map_to_update_period[module.weight] = \
                        map_to_update_period[module.weight].clamp(max=max_step).to(torch.int32)
                    
                    assert not torch.isnan(module.weight).any() 
                    assert not torch.isinf(module.weight).any()

                else:
                    last_weights[name] = this_weight
                    last_RQweights[name] = RQweight
        
    resume_model(model, optimizer, loss_scaler, args, device, args.opt_savepath)
    return map_to_update_period

def update_BSRamping_period(model, epoch, step, max_step, calib_data_loader,
                            optimizer, loss_scaler, max_norm, device, args, 
                            criterion, mixup_fn):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    torch.cuda.synchronize()
    
    calib_data_loader.sampler.set_epoch(epoch * 5000 + step)
    
    assert calib_data_loader is not None
    map_to_update_period = BSRamping_calib_update_period(model, 
                                        max_step=max_step,
                                        calib_data_loader=calib_data_loader, 
                                        optimizer=optimizer, 
                                        loss_scaler=loss_scaler,
                                        max_norm=max_norm,
                                        device=device,
                                        args=args,
                                        criterion = criterion,
                                        mixup_fn = mixup_fn
                                        )
    torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()
    
    # set optimizer to AdamW with element-wise BSRamping
    optimizer.reset_update_period(map_to_update_period) 
    print(f"end calibration of element-wise BSRamping: {start_event.elapsed_time(end_event) / 1000.0} s")
