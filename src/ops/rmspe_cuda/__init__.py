import torch
from torch.utils.cpp_extension import load

_ext = load(
    name="rmspe_ext",
    sources=[
        "src/ops/rmspe_cuda/rmspe.cpp",
        "src/ops/rmspe_cuda/rmspe_kernel.cu",
    ],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)

class RMSPEFromLogFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred_log, y_log, eps: float = 1e-8):
        ctx.save_for_backward(pred_log, y_log)
        ctx.eps = eps
        return _ext.mspe_forward(pred_log.contiguous(), y_log.contiguous(), float(eps))
    
    @staticmethod
    def backward(ctx, grad_out):
        pred_log, y_log = ctx.saved_tensors
        g = _ext.mspe_backward(pred_log.contiguous(), y_log.contiguous(), float(ctx.eps))
        return g * grad_out, None, None

def mspe_from_log_cuda(pred_log, y_log, eps: float = 1e-8):
    return RMSPEFromLogFn.apply(pred_log, y_log, eps)
