import torch

class SymmQuantizer_Torch(torch.autograd.function.InplaceFunction):
    @staticmethod
    def forward(ctx, input, bits, e_bit, stochastic, epsilon, apply_quantize=True, MXScale=0):
        with torch.no_grad():
            absmax_per_block = input.abs().amax(dim=(1, 2)).unsqueeze(1).unsqueeze(2)
            absmax_per_block[absmax_per_block == 0] += epsilon
            
            if bits == 100 or not apply_quantize:
                return input, input, torch.ones_like(absmax_per_block)
            elif bits == 32:
                return input.to(torch.float32), input.to(torch.float32), torch.ones_like(absmax_per_block)
            elif bits == 16:
                return input.to(torch.bfloat16), input.to(torch.bfloat16), torch.ones_like(absmax_per_block)
            else:
                ####### START --- floating point quantization #######
                
                assert e_bit < bits
                m_bit = bits - 1 - e_bit
                
                Elow = -2 ** (e_bit - 1) + 2        # `Elow` is FLEXIBLE, determining `BIAS`
                Ehigh = Elow + 2 ** e_bit - 2       # `Ehigh` depends on `Elow`
                Mhigh = 2 ** m_bit - 1
                
                Qmax = (1 + Mhigh / (Mhigh + 1)) * (2 ** Ehigh)
                Qmin = -Qmax
                
                # [exp]=0 =>      0.[mantissa] * 2^Elow
                # [exp]=1 =>      1.[mantissa] * 2^Elow
                # [exp]=2 =>  (2~3).[mantissa] * 2^Elow = 1.[mantissa] * 2^(Elow+1)
                # [exp]=3 =>  (4~7).[mantissa] * 2^Elow = 1.[mantissa] * 2^(Elow+2)
                def cast_to_fp_ExMy_with_stochastic(x, stochastic):
                    sign, x_abs = x.sign(), x.abs()
                    expo = torch.floor(torch.log2(x.abs() + epsilon))
                    expo = torch.clamp(expo, min=Elow)
                    mant = x_abs / (2 ** expo)
                    
                    mant_int = torch.floor(mant)
                    mant_frac = mant - mant_int
                    mant_frac = mant_frac * (Mhigh + 1)
                    
                    if stochastic:          # stochastic rounding
                        noise = mant_frac.new(mant_frac.shape).uniform_(-0.5, 0.5)
                        mant_frac.add_(noise)
                    mant_frac.clamp_(0, Mhigh + 1).round_()

                    mant_q = mant_int + mant_frac / (Mhigh + 1)
                    y = sign * (2 ** expo) * mant_q
                    
                    y.clamp_(Qmin, Qmax)    # deal with overflow and underflow
                                            # (caused by stochastic rounding)
                    return y
                
                scale_per_block = (2 * absmax_per_block) / (Qmax - Qmin)
                scale_per_block = scale_per_block.to(input)
                
                if MXScale == 1:    # TetraJet
                    scale_per_block = torch.ceil(torch.log2(scale_per_block)) 
                    scale_per_block.clamp_(-127, 127)
                    scale_per_block = 2 ** scale_per_block
                elif MXScale == 2:  # Wrong
                    scale_per_block = torch.floor(torch.log2(scale_per_block)) 
                    scale_per_block.clamp_(-127, 127)
                    scale_per_block = 2 ** scale_per_block
                elif MXScale == 3:  # Microscaling's original setting
                    scale_per_block = torch.floor(torch.log2(absmax_per_block)) - Ehigh 
                    scale_per_block.clamp_(-127, 127)
                    scale_per_block = 2 ** scale_per_block

                Qinput = cast_to_fp_ExMy_with_stochastic(input / scale_per_block, stochastic=stochastic)
                Qinput = Qinput.to(input)

                RQinput = Qinput * scale_per_block

                if input.dtype != Qinput.dtype:
                    import IPython
                    IPython.embed()
                    
                return RQinput, Qinput, scale_per_block
            
                ####### END   --- floating point quantization #######

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None


class SymmQuantizer_with_EMA_Torch(torch.autograd.function.InplaceFunction):
    @staticmethod
    def forward(ctx, input, 
                bits, e_bit, 
                epsilon, apply_quantize=True,
                ema_input = None, MXScale = 0):
        with torch.no_grad():
            absmax_per_block = input.abs().amax(dim=(1, 2)).unsqueeze(1).unsqueeze(2)
            absmax_per_block[absmax_per_block == 0] += epsilon
            
            if bits == 100 or not apply_quantize:
                return input, input, torch.ones_like(absmax_per_block)
            elif bits == 32:
                return input.to(torch.float32), input.to(torch.float32), torch.ones_like(absmax_per_block)
            elif bits == 16:
                return input.to(torch.bfloat16), input.to(torch.bfloat16), torch.ones_like(absmax_per_block)
            else:
                ####### START --- floating point quantization #######
                
                assert e_bit < bits
                m_bit = bits - 1 - e_bit
                
                Elow = -2 ** (e_bit - 1) + 2        # `Elow` is FLEXIBLE, determining `BIAS`
                Ehigh = Elow + 2 ** e_bit - 2       # `Ehigh` depends on `Elow`
                Mhigh = 2 ** m_bit - 1
                
                Qmax = (1 + Mhigh / (Mhigh + 1)) * (2 ** Ehigh)
                Qmin = -Qmax
                
                # print(f"\nQuantization: e{e_bit}m{m_bit}, Elow={Elow}, Qmin={Qmin}, Qmax={Qmax}")
                
                # [exp]=0 =>      0.[mantissa] * 2^Elow
                # [exp]=1 =>      1.[mantissa] * 2^Elow
                # [exp]=2 =>  (2~3).[mantissa] * 2^Elow = 1.[mantissa] * 2^(Elow+1)
                # [exp]=3 =>  (4~7).[mantissa] * 2^Elow = 1.[mantissa] * 2^(Elow+2)
                def cast_to_fp_ExMy_with_EMA(x, ema_x):
                    sign, x_abs = x.sign(), x.abs()
                    ema_x_signed = sign * ema_x                     # `ema_x`'s sign should "FOLLOW" x'sign
                    
                    expo = torch.floor(torch.log2(x.abs() + epsilon))
                    expo = torch.clamp(expo, min=Elow)
                    expo = 2 ** expo
                    
                    mant     = x_abs / expo
                    mant_ema = ema_x_signed / expo                  # `mant_ema` should "FOLLOW" x
                    
                    mant_int        = torch.floor(mant)
                    mant_frac       = (mant - mant_int)     * (Mhigh + 1)
                    mant_ema_frac   = (mant_ema - mant_int) * (Mhigh + 1)
                    
                    mant_frac.clamp_(0, Mhigh + 1).floor_()
                    mant_frac[mant_ema_frac >= mant_frac + 0.5] += 1  # Important Step!!
                                                                      # cast to integer according to mant_ema

                    mant_q = mant_int + mant_frac / (Mhigh + 1)
                    y = sign * expo * mant_q
                    
                    y.clamp_(Qmin, Qmax)    # deal with overflow and underflow
                                            # (caused by stochastic rounding)
                    return y
                
                scale_per_block = (2 * absmax_per_block) / (Qmax - Qmin)
                scale_per_block = scale_per_block.to(input)
                
                if MXScale == 1:    # TetraJet
                    scale_per_block = torch.ceil(torch.log2(scale_per_block)) 
                    scale_per_block.clamp_(-127, 127)
                    scale_per_block = 2 ** scale_per_block
                elif MXScale == 2:  # Wrong
                    scale_per_block = torch.floor(torch.log2(scale_per_block)) 
                    scale_per_block.clamp_(-127, 127)
                    scale_per_block = 2 ** scale_per_block
                elif MXScale == 3:  # Microscaling's original setting
                    scale_per_block = torch.floor(torch.log2(absmax_per_block)) - Ehigh 
                    scale_per_block.clamp_(-127, 127)
                    scale_per_block = 2 ** scale_per_block

                Qinput = cast_to_fp_ExMy_with_EMA(
                    input       / scale_per_block,
                    ema_input   / scale_per_block
                )
                Qinput = Qinput.to(input)
                RQinput = Qinput * scale_per_block

                if input.dtype != Qinput.dtype:
                    import IPython
                    IPython.embed()
                    
                return RQinput, Qinput, scale_per_block
            
                ####### END   --- floating point quantization #######

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None
