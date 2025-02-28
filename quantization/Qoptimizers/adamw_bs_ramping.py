""" Impl based on PyTorch AdamW Optimizer
"""
import math
import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional
from torch import Tensor

class AdamW_BS_Ramping(Optimizer):
    r"""
    Based on AdamW in torch.optim lib
    """
    
    def __init__(self, params, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, *, maximize: bool = False,
                 foreach: Optional[bool] = None,
                 capturable: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        foreach=foreach, maximize=maximize, capturable=capturable)
        super().__init__(params, defaults)
        
        self.is_linear = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"frozen Parameters : {name}")
                continue  # frozen weights
            
            is_linear = ("mlp.fc" in name or "attn.proj" in name or "attn.qkv" in name) and (".weight" in name)
            self.is_linear[param] = is_linear
    
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))
    
    @torch.no_grad()
    def clear_for_ramping(self, map_to_update_period = None):
        self.map_to_update_period = {} if map_to_update_period is None else map_to_update_period
        
        for group in self.param_groups:
            for p in group['params']:
                # State initialization
                if p in self.state:
                    state = self.state[p]
                    state['update_period'] = self.map_to_update_period[p] \
                        if p in self.map_to_update_period \
                        else torch.ones_like(p, memory_format=torch.preserve_format)
                    state['step_acc'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['grad_acc'] = torch.zeros_like(p, memory_format=torch.preserve_format)
    
    @torch.no_grad()
    def reset_update_period(self, map_to_update_period = None):
        self.map_to_update_period = {} if map_to_update_period is None else map_to_update_period
        for group in self.param_groups:
            for p in group['params']:
                # State initialization
                if p in self.state:
                    state = self.state[p]
                    state['update_period'] = self.map_to_update_period[p] \
                        if p in self.map_to_update_period \
                        else torch.ones_like(p, memory_format=torch.preserve_format)

                    if 'grad_acc' not in state:
                        state['step_acc'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['grad_acc'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
        
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            grads_acc = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            
            # ========= New List for elementwise batchsize-ramping =======
            update_periods = []
            step_accs = []

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                    # ========= New State Key for elementwise batchsize-ramping =======
                    state['update_period'] = self.map_to_update_period[p] \
                        if p in self.map_to_update_period \
                        else torch.ones_like(p, memory_format=torch.preserve_format)
                    state['step_acc'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['grad_acc'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                grads_acc   .append(state['grad_acc'])
                exp_avgs    .append(state['exp_avg'])
                exp_avg_sqs .append(state['exp_avg_sq'])
                update_periods.append(state['update_period'])
                step_accs  .append(state['step_acc'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                state_steps.append(state['step'])

            adamw_ElementWise_BSRamping(params_with_grad,
                  grads,
                  grads_acc, # added for BSRamping
                  exp_avgs,
                  exp_avg_sqs,
                  max_exp_avg_sqs,
                  state_steps,
                  update_periods= update_periods, # added for BSRamping
                  step_accs  = step_accs,   # added for BSRamping
                  amsgrad=amsgrad,
                  beta1=beta1,
                  beta2=beta2,
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  eps=group['eps'],
                  maximize=group['maximize'],
                  foreach=group['foreach'],
                  capturable=group['capturable'])

        return loss



def adamw_ElementWise_BSRamping(params: List[Tensor],
          grads: List[Tensor],
          grads_acc: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          max_exp_avg_sqs: List[Tensor],
          state_steps: List[Tensor],
          update_periods: List[Tensor],
          step_accs: List[Tensor],
          # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
          # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
          foreach: bool = None,
          capturable: bool = False,
          *,
          amsgrad: bool,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float,
          maximize: bool):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adamwNew
    else:
        func = _single_tensor_adamwNew

    func(params,
         grads,
         grads_acc,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps,
         update_periods,
         step_accs,
         amsgrad=amsgrad,
         beta1=beta1,
         beta2=beta2,
         lr=lr,
         weight_decay=weight_decay,
         eps=eps,
         maximize=maximize,
         capturable=capturable)


def _single_tensor_adamwNew(params: List[Tensor],
                         grads: List[Tensor],
                         grads_acc: List[Tensor],
                         exp_avgs: List[Tensor],
                         exp_avg_sqs: List[Tensor],
                         max_exp_avg_sqs: List[Tensor],
                         state_steps: List[Tensor],
                         update_periods: List[Tensor],
                         step_accs: List[Tensor],
                         *,
                         amsgrad: bool,
                         beta1: float,
                         beta2: float,
                         lr: float,
                         weight_decay: float,
                         eps: float,
                         maximize: bool,
                         capturable: bool):

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        grad_acc = grads_acc[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        
        # element-wise BS Ramping
        update_period = update_periods[i]
        step_acc = step_accs[i]
        step_acc += 1
        if not torch.isinf(grad).any() and not torch.isnan(grad).any():
            grad_acc += grad
        
        # `update_period` may change over time, 
        #  so `step_acc` could be larger than update_period
        update_mask = step_acc % update_period == 0
        
        if capturable:
            assert param.is_cuda and step_t.is_cuda, "If capturable=True, params and state_steps must be CUDA tensors."

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform weight decay
        # NOTE: weight_decay is for the whole tensor
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        t_grad = grad_acc[update_mask] / step_acc[update_mask]  # batchsize ramping
        exp_avg   [update_mask] = exp_avg   [update_mask] * beta1 + t_grad * (1 - beta1)
        exp_avg_sq[update_mask] = exp_avg_sq[update_mask] * beta2 + (t_grad * t_grad) * (1 - beta2)

        if capturable:
            raise NotImplementedError
        else:
            step = step_t.item()

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            
            new_lr = lr * step_acc # a tensor with shape like `param`
            step_size = new_lr / bias_correction1

            bias_correction2_sqrt = math.sqrt(bias_correction2)

            if amsgrad:
                raise NotImplementedError
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            # Perform update according to momentum
            param[update_mask] -= (exp_avg / denom * step_size)[update_mask]

        step_acc[update_mask] = 0
        grad_acc[update_mask] = 0


def _multi_tensor_adamwNew(params: List[Tensor],
                        grads: List[Tensor],
                        grads_acc: List[Tensor],
                        exp_avgs: List[Tensor],
                        exp_avg_sqs: List[Tensor],
                        max_exp_avg_sqs: List[Tensor],
                        state_steps: List[Tensor],
                        update_periods: List[Tensor],
                        step_accs: List[Tensor],
                        *,
                        amsgrad: bool,
                        beta1: float,
                        beta2: float,
                        lr: float,
                        weight_decay: float,
                        eps: float,
                        maximize: bool,
                        capturable: bool):
    raise NotImplementedError