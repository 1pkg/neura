from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, UpSampling2D

from .base import Base
from .loss_func import psnr_loss


class ConvolutionDeep(Base):
    def _model(self):
        model = Sequential()
        model.add(
            Conv2D(
                128,
                9,
                padding='same',
                activation='relu',
                input_shape=(160, 90, 3),
            ))  # 1
        model.add(
            Conv2D(
                128,
                3,
                padding='same',
                activation='relu',
                input_shape=(160, 90, 128),
            ))  # 2
        model.add(
            Conv2D(
                128,
                5,
                padding='same',
                activation='relu',
                input_shape=(160, 90, 128),
            ))  # 3
        model.add(UpSampling2D())
        model.add(
            Conv2D(
                64,
                9,
                padding='same',
                activation='relu',
                input_shape=(320, 180, 128),
            ))  # 4
        model.add(
            Conv2D(
                64,
                3,
                padding='same',
                activation='relu',
                input_shape=(320, 180, 64),
            ))  # 5
        model.add(
            Conv2D(
                64,
                5,
                padding='same',
                activation='relu',
                input_shape=(320, 180, 64),
            ))  # 6
        model.add(UpSampling2D())
        model.add(
            Conv2D(
                32,
                9,
                padding='same',
                activation='relu',
                input_shape=(640, 360, 64),
            ))  # 7
        model.add(
            Conv2D(
                32,
                3,
                padding='same',
                activation='relu',
                input_shape=(640, 360, 32),
            ))  # 8
        model.add(
            Conv2D(
                32,
                5,
                padding='same',
                activation='relu',
                input_shape=(640, 360, 32),
            ))  # 9
        model.add(
            Conv2D(
                3,
                9,
                padding='same',
                activation='relu',
                input_shape=(640, 360, 3),
            ))  # 10
        model.compile(
            optimizer=Adam(lr=1e-3),
            loss='mse',
            metrics=[psnr_loss],
        )
        return model