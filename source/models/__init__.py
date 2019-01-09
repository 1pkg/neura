from .convolution import Convolution
from .convolution_avg import ConvolutionAvg
from .convolution_denoise import ConvolutionDenoise
from .convolution_rec import ConvolutionRec
from .convolution_scaled import ConvolutionScaled

from .interpolation_bicubic import InterpolationBicubic
from .interpolation_bilenear import InterpolationBilenear
from .interpolation_lancoz import InterpolationLancoz
from .interpolation_nearest import InterpolationNearest

__all__ = [
    'Convolution',
    'ConvolutionAvg',
    'ConvolutionDenoise',
    'ConvolutionRec',
    'ConvolutionScaled',
    # fake interpolation models
    'InterpolationBicubic',
    'InterpolationBilenear',
    'InterpolationLancoz',
    'InterpolationNearest',
]