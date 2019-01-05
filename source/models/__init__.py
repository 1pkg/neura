from .convolution_deep import ConvolutionDeep
from .convolution_merge import ConvolutionMerge
from .convolution_shallow import ConvolutionShallow
from .convolution_shallow_scaled import ConvolutionShallowScaled

from .interpolation_bicubic import InterpolationBicubic
from .interpolation_bilenear import InterpolationBilenear
from .interpolation_nearest import InterpolationNearest

__all__ = [
    'ConvolutionDeep',
    'ConvolutionMerge',
    'ConvolutionShallow',
    'ConvolutionShallowScaled',
    # fake interpolation models
    'InterpolationBicubic',
    'InterpolationBilenear',
    'InterpolationNearest',
]