from PIL import Image

from .base_interpolation import BaseInterpolation


class InterpolationBicubic(BaseInterpolation):
    _scale_type = Image.BICUBIC