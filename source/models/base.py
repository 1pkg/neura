from os import path
from keras.models import load_model

from utils import Loader, Keeper


class Base:
    def __init__(self):
        self.__path_base = path.join(
            path.dirname(__file__),
            '..',
            '..',
            'dump',
            'models',
        )
        self.__model = None

    def name(self):
        return self.__class__.__name__.lower()

    def model(self):
        if self.__model is None:
            self.__model = self._model()
        return self.__model

    def _model(self):
        return NotImplemented

    def serialize(self):
        path_model = path.join(self.__path_base, self.name() + '.h5')
        self.model().save(path_model)

    def unserialize(self):
        path_model = path.join(self.__path_base, self.name() + '.h5')
        if path.isfile(path_model):
            self.__model = load_model(path_model)

    def train(
            self,
            bundle,
            bundle_size=32,
            steps_per_epoch=128,
            epochs=8,
            verbose=True,
    ):
        loader = Loader(bundle)
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
    ):
        loader = Loader(bundle)
        return self.model().evaluate_generator(
            loader.gen_data_test(bundle_size),
            steps=1,
            verbose=verbose,
        )

    def test(self, bundle, bundle_size=32, verbose=True):
        loader = Loader(bundle)
        data = loader.data_test(bundle_size)
        result = self.model().predict(
            data[0],
            batch_size=bundle_size,
            verbose=verbose,
        )

        keeper = Keeper(bundle, self.name())
        for i in range(0, len(data[0])):
            keeper.save(f'{bundle}-{i}', data[1][i], data[0][i], result[i])