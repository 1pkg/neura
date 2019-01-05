from PIL import Image

from .base_interpolation import BaseInterpolation


class InterpolationNearest(BaseInterpolation):
    _scale_type = Image.NEAREST