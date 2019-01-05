from PIL import Image

from .base_interpolation import BaseInterpolation


class InterpolationBilenear(BaseInterpolation):
    _scale_type = Image.BILINEAR