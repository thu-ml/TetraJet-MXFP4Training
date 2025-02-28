# TetraJet: Oscillation-Reduced MXFP4 Training for Vision Transformers

This repository provides the official implementation of "Oscillation-Reduced MXFP4 Training for Vision Transformers" on DeiT model pre-training, which enable a more accurate MXFP4 pre-training for Vision Transformers.

This repository is adapted from open-source repo [deit](https://github.com/facebookresearch/deit) by facebookresearch. 

## Getting Started

Environment:

- `CUDA==11.7`, `Python>=3.8`
- `triton==3.0.0`
- `torch==1.13.1+cu117, torchaudio==0.13.1+cu117, torchvision==0.14.1+cu117`
- `numpy`, `nvidia-pyindex`, `nvidia-dllogger`, `tensorboardX`

Usage: 

- Prepare ImageNet2012 Dataset according to [README_deit.md](README_deit.md).
- Run the scripts in `scripts` to reproduce pre-training results.

## Repository Overview

- `quantization/`: 
  - Fine-grained quantization to low-precision floating-point.
  - Forward and backward process in MXFP4 format for Linear layers.
  - Customized quantization method (Q-EMA) & optimizer (Q-Ramping) for a more stable low-precision training.
- `timm/`: We modified the original `timm` source codes. 
  - Replace all the linear layers in Transformer blocks in `timm/models/vision_transformer.py`.
  - Add support for our Q-Ramping optimizer.
- `scripts/`: scripts to reproduce results.

Due to hardware limitation, we only provide the simulation codes of MXFP4 fully-quantized training, which can run on most GPUs. 

## Citation

TODO