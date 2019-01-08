from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D

from .base_scaled import BaseScaled
from .loss_func import psnr_loss


class ConvolutionScaled(BaseScaled):
    def _model(self):
        model = Sequential()
        model.add(
            Conv2D(
                64,
                9,
                padding='same',
                activation='relu',
                input_shape=(640, 360, 3),
            ))
        model.add(
            Conv2D(
                32,
                1,
                padding='same',
                activation='relu',
                input_shape=(640, 360, 64),
            ))
        model.add(
            Conv2D(
                3,
                5,
                padding='same',
                activation='relu',
                input_shape=(640, 360, 32),
            ))
        model.compile(
            optimizer=Adam(lr=1e-3),
            loss='mse',
            metrics=[psnr_loss],
        )
        return model