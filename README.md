# TetraJet: Oscillation-Reduced MXFP4 Training

This repo provides the official implementation of "Oscillation-Reduced MXFP4 Training for Vision Transformers".

**Oscillation-Reduced MXFP4 Training for Vision Transformers**  
Paper: https://arxiv.org/abs/2502.20853

> **Abstract**  
> Pre-training Transformers in FP4 precision is becoming a promising approach to gain substantial speedup, but it comes with a considerable loss of accuracy. Microscaling (MX) data format provides a fine-grained per-group quantization method to improve the representation ability of the FP4 format and is supported by the next-generation Blackwell GPU architecture. However, training with MXFP4 data format still results in significant degradation and there is a lack of systematic research on the reason.
>
> In this work, we propose a novel training method TetraJet for a more accurate FP4 training. We comprehensively evaluate all of the quantizers involved in the training, and identify the weight oscillation problem in the forward pass as the main source of the degradation in MXFP4 training. Therefore, we introduce two novel methods, EMA Quantizer (Q-EMA) and Adaptive Ramping Optimizer (Q-Ramping), to resolve the oscillation problem. Extensive experiments on Vision Transformers demonstrate that TetraJet consistently outperforms the existing 4-bit training methods, and Q-EMA & Q-Ramping can provide additional enhancement by effectively reducing oscillation. We decreased the accuracy degradation by more than 50% compared to the baseline, and can even achieve competitive performance compared to full precision training.

This repository is adapted from the open-source repo [deit](https://github.com/facebookresearch/deit) by facebookresearch. 

## Getting Started

Environment:

- `CUDA==11.7`, `Python>=3.8`
- `triton==3.0.0`
- `torch==1.13.1+cu117`, `torchaudio==0.13.1+cu117`, `torchvision==0.14.1+cu117`
- `numpy`, `nvidia-pyindex`, `nvidia-dllogger`, `tensorboardX`

Usage: 

- Prepare ImageNet2012 Dataset according to [README_deit.md](https://github.com/facebookresearch/deit/blob/main/README_deit.md#data-preparation).
- Run the scripts in `scripts` to reproduce pre-training results.

## Repository Overview

- `quantization/`: 
  - Fine-grained quantization to low-precision floating-point.
  - Forward and backward process in MXFP4 format for Linear layers.
  - Customized quantization method (Q-EMA) & optimizer (Q-Ramping) for a more stable low-precision training.
- `timm/`: We modified the original [timm](https://github.com/huggingface/pytorch-image-models) source codes. 
  - Replace all the linear layers in Transformer blocks in `timm/models/vision_transformer.py`.
  - Add support for our Q-Ramping optimizer.
- `scripts/`: scripts to reproduce results.

Due to hardware limitations, we only provide the simulation codes of MXFP4 fully-quantized training, which can run on most GPUs. 

## Citation

If you find this work useful, please consider citing:
```bibtex
@article{chen2025oscillation,
  title={Oscillation-Reduced MXFP4 Training for Vision Transformers},
  author={Chen, Yuxiang and Xi, Haocheng and Zhu, Jun and Chen, Jianfei},
  journal={arXiv preprint arXiv:2502.20853},
  year={2025}
}
```
