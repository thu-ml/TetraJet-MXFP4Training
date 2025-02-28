# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
import torch.distributed as dist
import utils
from pathlib import Path

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import dllogger as DLLogger

from quantization import update_BSRamping_period, QLinearLayer

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, eval_data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None,
                    calib_data_loader: Optional[Iterable] = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    if "BSRamping" in args.opt:
        if args.opt_ramping_type <= 5:
            CALIBRATE_PERIOD = len(data_loader)
        else:
            raise NotImplementedError
        
        max_step = CALIBRATE_PERIOD
    
    output_dir = Path(args.output_dir)
    
    for step, (samples, targets) in metric_logger.log_every(data_loader, print_freq, header):
        if "BSRamping" in args.opt and step % CALIBRATE_PERIOD == 0:
            assert max_norm is None, "Only Support no-grad_clipping !!"
            print(f"start calibration of element-wise BSRamping at [epoch#{epoch} step#{step}/{len(data_loader)}]!!!")
            update_BSRamping_period(model, epoch=epoch, step=step, max_step=max_step, calib_data_loader=calib_data_loader,
                                    optimizer=optimizer, loss_scaler=loss_scaler, max_norm=max_norm, device=device, args=args, 
                                    criterion=criterion, mixup_fn=mixup_fn)
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(samples)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        
        if args.qlinear_ema_decay > 0:         # EMA-Weight Update
            for name, module in model.module.named_modules():
                if isinstance(module, QLinearLayer):
                    if module.training and module.apply_quantize and module.is_ema:
                        module.ema_step += 1
                        module.ema_weight.mul_(module.ema_decay).add_(module.weight.data, alpha = 1 - module.ema_decay)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        step_summary = {
            "loss": loss.item(),
        }
        DLLogger.log(step=epoch * len(data_loader) + step,
                     data=step_summary, verbosity=0)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device,
             copied_data_num=0): 
    """ 
    copied_data_num:    ignore results from [-copied_data_num:] 
    
    NOTE: because we copied `copied_data_num` duplicate samples to make 
                len(dataset_val) % batch_size == 0
          in acc calculation, we should ignore them """
    
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    unloaded_num = len(data_loader.dataset)
    for step, (images, target) in metric_logger.log_every(data_loader, 100, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        batch_size = target.shape[0]
        unloaded_num -= batch_size
        if unloaded_num == 0 and copied_data_num > 0:
            target = target[:-copied_data_num]
            batch_size = target.shape[0]
            print(f"eval cut : batch_size = {batch_size}")

        # compute output
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = model(images)
            
            if unloaded_num == 0 and copied_data_num > 0:
                output = output[:batch_size]
            loss = criterion(output, target)
      
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
