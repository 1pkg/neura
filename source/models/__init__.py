from .convolution_avg import ConvolutionAvg
from .convolution_deconvolution import ConvolutionDeconvolution
from .convolution_deep import ConvolutionDeep
from .convolution_shallow import ConvolutionShallow
from .convolution_shallow_scaled import ConvolutionShallowScaled

from .interpolation_bicubic import InterpolationBicubic
from .interpolation_bilenear import InterpolationBilenear
from .interpolation_lancoz import InterpolationLancoz
from .interpolation_nearest import InterpolationNearest

__all__ = [
    'ConvolutionAvg',
    'ConvolutionDeconvolution',
    'ConvolutionDeep',
    'ConvolutionShallow',
    'ConvolutionShallowScaled',
    # fake interpolation models
    'InterpolationBicubic',
    'InterpolationBilenear',
    'InterpolationLancoz',
    'InterpolationNearest',
]