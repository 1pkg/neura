from os import path
from inflection import underscore
from keras.models import load_model
from pickle import dump, load

from .loss_func import psnr_loss
from utils import Loader, Keeper


class Base:
    def __init__(self):
        self._path_model = path.join(
            path.dirname(__file__),
            '..',
            '..',
            'dump',
            'model',
        )
        self._path_history = path.join(
            path.dirname(__file__),
            '..',
            '..',
            'dump',
            'history',
        )
        self.__model = None

    def name(self):
        return underscore(self.__class__.__name__)

    def model(self):
        if self.__model is None:
            self.__model = self._model()
        return self.__model

    def _model(self):
        return NotImplemented

    def serialize(self):
        path_model = path.join(self._path_model, self.name() + '.h5')
        self.model().save(path_model)

    def unserialize(self):
        path_model = path.join(self._path_model, self.name() + '.h5')
        if path.isfile(path_model):
            self.__model = load_model(
                path_model,
                custom_objects={'psnr_loss': psnr_loss},
            )

    def history(self):
        path_history = path.join(self._path_history, self.name() + '.hi')
        with open(path_history, 'rb') as file_hi:
            return load(file_hi)

    def train(
            self,
            bundle,
            bundle_size=32,
            steps_per_epoch=128,
            epochs=8,
            verbose=True,
    ):
        loader = Loader(bundle)
        history = self.model().fit_generator(
            loader.gen_data_train(bundle_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
            validation_data=loader.gen_data_test(bundle_size),
            validation_steps=steps_per_epoch,
        )
        path_history = path.join(self._path_history, self.name() + '.hi')
        with open(path_history, 'wb') as file_hi:
            dump(history.history, file_hi)

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

    def test(
            self,
            bundle,
            bundle_size=32,
            verbose=True,
    ):
        loader = Loader(bundle)
        data = loader.data_test(bundle_size)
        result = self.model().predict(
            data[0],
            batch_size=bundle_size,
            verbose=verbose,
        )

        keeper = Keeper(bundle, self.name())
        for i in range(0, len(data[0])):
            keeper.save(f'{bundle}-{i+1}', data[1][i], data[0][i], result[i])