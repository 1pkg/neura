from keras.models import Sequential
from keras.layers import Conv2D, Dropout

from .base_scaled import BaseScaled
from .loss_func import psnr_loss


class ConvolutionShallowScaled(BaseScaled):
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
        model.add(Dropout(0.15))
        model.compile(
            loss='mse',
            optimizer='adam',
            metrics=['mse', psnr_loss],
        )
        return model