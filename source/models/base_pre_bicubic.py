from .base import Base
from utils import Loader, Keeper


class BasePreBicubic(Base):
    def train(
            self,
            bundle,
            bundle_size=32,
            steps_per_epoch=128,
            epochs=8,
            verbose=True,
            scale_factor=4,
    ):
        loader = Loader(bundle, scale_factor, 3)
        self.model().fit_generator(
            loader.gen_data_train(bundle_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
        )

    def score(
            self,
            bundle,
            bundle_size=32,
            verbose=True,
            scale_factor=4,
    ):
        loader = Loader(bundle, scale_factor, 3)
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
        loader = Loader(bundle, scale_factor, 3)
        data = loader.data_test(bundle_size)
        result = self.model().predict(
            data[0],
            batch_size=bundle_size,
            verbose=verbose,
        )

        keeper = Keeper(bundle, self.name())
        for i in range(0, len(data[0])):
            keeper.save(f'{bundle}-{i}', data[1][i], data[0][i], result[i])