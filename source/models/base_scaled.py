from os import path
from PIL import Image
from pickle import dump, load

from .base import Base
from .loss_func import psnr_loss
from utils import Loader, Keeper


class BaseScaled(Base):
    def train(
            self,
            bundle,
            bundle_size=32,
            steps_per_epoch=128,
            epochs=8,
            verbose=True,
            scale_factor=4,
    ):
        loader = Loader(bundle, scale_factor, Image.BICUBIC)
        history = self.model().fit_generator(
            loader.gen_data_train(bundle_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
        )
        path_history = path.join(self._path_history, self.name() + '.hi')
        with open(path_history, 'wb') as file_hi:
            dump(history.history, file_hi)

    def score(
            self,
            bundle,
            bundle_size=32,
            verbose=True,
            scale_factor=4,
    ):
        loader = Loader(bundle, scale_factor, Image.BICUBIC)
        return self.model().evaluate_generator(
            loader.gen_data_test(bundle_size),
            steps=1,
            verbose=verbose,
        )

    def test(
            self,
            bundle,
            bundle_size=32,
            verbose=True,
            scale_factor=4,
    ):
        loader = Loader(bundle, scale_factor, Image.BICUBIC)
        data = loader.data_test(bundle_size)
        result = self.model().predict(
            data[0],
            batch_size=bundle_size,
            verbose=verbose,
        )

        keeper = Keeper(bundle, self.name())
        for i in range(0, len(data[0])):
            keeper.save(f'{bundle}-{i+1}', data[1][i], data[0][i], result[i])