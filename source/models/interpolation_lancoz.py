from PIL import Image

from .base_interpolation import BaseInterpolation


class InterpolationLancoz(BaseInterpolation):
    _scale_type = Image.LANCZOS