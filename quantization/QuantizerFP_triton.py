import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

class SymmQuantizer_Triton(torch.autograd.function.InplaceFunction):
    @staticmethod
    def forward(ctx, input, bits, e_bit, stochastic, epsilon, apply_quantize=True, MXScale = 0):
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
                
                Qinput = floatExMy_quantize_triton(input / scale_per_block, 
                                                   e_bit, m_bit, stochastic=stochastic).clamp(Qmin, Qmax) # deal with overflow and underflow
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


class SymmQuantizer_with_EMA_Triton(torch.autograd.function.InplaceFunction):
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

                Qinput = floatExMy_quantizeEMA_triton(
                    input       / scale_per_block,
                    ema_input   / scale_per_block,
                    e_bit, m_bit
                ).clamp(Qmin, Qmax) # deal with overflow and underflow
                
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


def floatExMy_quantize_triton(x, e_bit, m_bit, stochastic):
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    y = torch.zeros_like(x)

    if x.dtype in [torch.bfloat16, torch.float32]:
        if stochastic:
            noise = x.new(x.shape).uniform_(-0.5, 0.5)
            _floatExMy_stochastic_quantize_kernel[grid](x, noise, y, n_elements, e_bit, m_bit)
        else:
            _floatExMy_quantize_kernel[grid](x, y, n_elements, e_bit, m_bit)
    else:
        raise NotImplementedError(f"Other data format {x.dtype} for float quantization triton")

    return y

def floatExMy_quantizeEMA_triton(x, x_ema, e_bit, m_bit):
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    y = torch.zeros_like(x)

    if x.dtype in [torch.bfloat16, torch.float32]:
        _floatExMy_quantizeEMA_kernel[grid](x, x_ema, y, n_elements, e_bit, m_bit)
    else:
        raise NotImplementedError(f"Other data format {x.dtype} for float quantization triton")

    return y

@triton.autotune(
        configs=[
            # triton.Config({'BLOCK_SIZE': 4,}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 4096,}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 2048,}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 2048,}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 1024,}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 2048,}, num_stages=2),
            triton.Config({'BLOCK_SIZE': 2048,}, num_stages=1),
        ],
        key=['n_elements']
)
@triton.jit
def _floatExMy_quantize_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    e_bit, m_bit,
    BLOCK_SIZE: tl.constexpr,
):
    if isinstance(e_bit, tl.constexpr):
        ebit = e_bit.value
    else:
        ebit = e_bit

    if isinstance(m_bit, tl.constexpr):
        mbit = m_bit.value
    else:
        mbit = m_bit
    
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    x = x.to(tl.float32)
    sign = 1 - 2 * libdevice.signbit(x)
    x_abs = tl.abs(x)
    Elow = -tl.exp2((ebit - 1).to(tl.float32)) + 2
    Ehigh = tl.exp2((ebit - 1).to(tl.float32))
    Mhigh = tl.exp2(mbit.to(tl.float32))
    expo = tl.floor(tl.log2(x_abs))
    expo = tl.clamp(expo, min=Elow, max=Ehigh)
    mant = x_abs / tl.exp2(expo)

    mant_int = tl.floor(mant)
    mant_frac = mant - mant_int
    mant_frac = mant_frac * Mhigh
    # mant_frac = mant_frac + noise
    mant_frac = libdevice.round(mant_frac)

    mant_q = mant_int + mant_frac / Mhigh
    y = sign * tl.exp2(expo) * mant_q
    y = y.to(x_ptr.dtype.element_ty)

    tl.store(output_ptr + offsets, y, mask=mask)


@triton.autotune(
        configs=[
            # triton.Config({'BLOCK_SIZE': 4,}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 4096,}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 2048,}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 2048,}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 1024,}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 2048,}, num_stages=2),
            triton.Config({'BLOCK_SIZE': 2048,}, num_stages=1),
        ],
        key=['n_elements']
)
@triton.jit
def _floatExMy_stochastic_quantize_kernel(
    x_ptr,
    noise_ptr,
    output_ptr,
    n_elements,
    e_bit,
    m_bit,
    BLOCK_SIZE: tl.constexpr,
):
    if isinstance(e_bit, tl.constexpr):
        ebit = e_bit.value
    else:
        ebit = e_bit

    if isinstance(m_bit, tl.constexpr):
        mbit = m_bit.value
    else:
        mbit = m_bit
    
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    noise = tl.load(noise_ptr + offsets, mask=mask)
    
    x = x.to(tl.float32)
    sign = 1 - 2 * libdevice.signbit(x)
    x_abs = tl.abs(x)
    Elow = -tl.exp2((ebit - 1).to(tl.float32)) + 2
    Ehigh = tl.exp2((ebit - 1).to(tl.float32))
    Mhigh = tl.exp2(mbit.to(tl.float32))
    expo = tl.floor(tl.log2(x_abs))
    expo = tl.clamp(expo, min=Elow, max=Ehigh)
    mant = x_abs / tl.exp2(expo)

    mant_int = tl.floor(mant)
    mant_frac = mant - mant_int
    mant_frac = mant_frac * Mhigh
    mant_frac = mant_frac + noise
    mant_frac = libdevice.round(mant_frac)

    mant_q = mant_int + mant_frac / Mhigh
    y = sign * tl.exp2(expo) * mant_q
    y = y.to(x_ptr.dtype.element_ty)

    tl.store(output_ptr + offsets, y, mask=mask)
    

@triton.autotune(
        configs=[
            # triton.Config({'BLOCK_SIZE': 4,}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 4096,}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 2048,}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 2048,}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 1024,}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 2048,}, num_stages=2),
            triton.Config({'BLOCK_SIZE': 2048,}, num_stages=1),
        ],
        key=['n_elements']
)
@triton.jit
def _floatExMy_quantizeEMA_kernel(
    x_ptr,
    x_ema_ptr,
    output_ptr,
    n_elements,
    e_bit,
    m_bit,
    BLOCK_SIZE: tl.constexpr,
):
    if isinstance(e_bit, tl.constexpr):
        ebit = e_bit.value
    else:
        ebit = e_bit

    if isinstance(m_bit, tl.constexpr):
        mbit = m_bit.value
    else:
        mbit = m_bit
    
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # read tensor pattern
    x     = tl.load(x_ptr     + offsets, mask=mask)
    x_ema = tl.load(x_ema_ptr + offsets, mask=mask)
    
    x     = x    .to(tl.float32)
    x_ema = x_ema.to(tl.float32)
    
    sign = 1 - 2 * libdevice.signbit(x)
    x_abs = tl.abs(x)
    x_ema_signed = sign * x_ema
    
    Elow = -tl.exp2((ebit - 1).to(tl.float32)) + 2
    Ehigh = tl.exp2((ebit - 1).to(tl.float32))
    Mhigh = tl.exp2(mbit.to(tl.float32))
    expo = tl.floor(tl.log2(x_abs))
    expo = tl.clamp(expo, min=Elow, max=Ehigh)
    
    mant     = x_abs / tl.exp2(expo)
    mant_ema = x_ema_signed / tl.exp2(expo)

    mant_int = tl.floor(mant)
    mant_frac = mant - mant_int
    mant_frac = mant_frac * Mhigh
    mant_frac = libdevice.floor(mant_frac)
    
    mant_ema_frac = mant_ema - mant_int
    mant_ema_frac = mant_ema_frac * Mhigh
    update_mask = mant_ema_frac >= mant_frac + 0.5
    mant_frac = tl.where(update_mask, mant_frac + 1, mant_frac)

    mant_q = mant_int + mant_frac / Mhigh
    y = sign * tl.exp2(expo) * mant_q
    y = y.to(x_ptr.dtype.element_ty)

    tl.store(output_ptr + offsets, y, mask=mask)
    
