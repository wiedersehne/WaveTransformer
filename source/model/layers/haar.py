# NOTE - not used - would ideally add this in the final version and then train end-to-end on the full sequences data

from torch.autograd import Function
import torch
import torch.nn as nn
import pywt
import numpy as np
from torch.nn.parameter import Parameter


class HaarIWTFunction(Function):

    @staticmethod
    def forward(ctx, input):
        numpy_input = input.detach().numpy()
        result = pywt.wavedec(numpy_input, pywt.Wavelet("haar"))
        ctx.save_for_backward(input)
        return torch.as_tensor(result, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):

        numpy_go = grad_output.numpy()

        result = irfft2(numpy_go)
        return grad_output.new(result)


# since this layer does not have any parameters, we can
# simply declare this as a function, rather than as an nn.Module class
def haar_iwt(input):
    return HaarIWTFunction.apply(input)
