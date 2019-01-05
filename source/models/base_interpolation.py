from PIL import Image
import numpy

from .base import Base
from .loss_func import psnr_np, mse_np
from utils import Loader, Keeper


class BaseInterpolation(Base):
    _scale_type = None
    _scale_factor = 4

    def model(self):
        return None

    def serialize(self):
        pass

    def unserialize(self):
        pass

    def history(self):
        pass

    def train(
            self,
            bundle,
            bundle_size=32,
            steps_per_epoch=128,
            epochs=8,
            verbose=True,
    ):
        pass

    def score(
            self,
            bundle,
            bundle_size=32,
            verbose=True,
    ):
        loader = Loader(bundle)
        data = loader.data_test(bundle_size)
        averages_mse = 0.0
        averages_psnr = 0.0
        for i in range(0, len(data[0])):
            image = numpy.array(data[0][i])
            image *= 255
            image = Image.fromarray(numpy.uint8(image))
            image = image.convert('RGB')
            image = image.resize((
                int(image.size[0] * self._scale_factor),
                int(image.size[1] * self._scale_factor),
            ), self._scale_type)
            image = numpy.array(image)
            image = image.astype('float32')
            image /= 255
            averages_mse += mse_np(data[1][i], image)
            averages_psnr += psnr_np(data[1][i], image)
        return (
            averages_mse / len(data[0]),
            averages_psnr / len(data[0]),
        )

    def test(
            self,
            bundle,
            bundle_size=32,
            verbose=True,
    ):
        loader = Loader(bundle)
        data = loader.data_test(bundle_size)
        result = []
        for image in data[0]:
            image = numpy.array(data[0][i])
            image *= 255
            image = Image.fromarray(numpy.uint8(image))
            image = image.convert('RGB')
            image = image.resize((
                int(image.size[0] * self._scale_factor),
                int(image.size[1] * self._scale_factor),
            ), self._scale_type)
            image = numpy.array(image)
            image = image.astype('float32')
            image /= 255
            result.append(image)

        keeper = Keeper(bundle, self.name())
        for i in range(0, len(data[0])):
            keeper.save(f'{bundle}-{i+1}', data[1][i], data[0][i], result[i])