from keras.models import Sequential
from keras.layers import Conv2D, Dropout, UpSampling2D

from .base import Base
from .loss_func import psnr_loss


class ConvolutionDeep(Base):
    def _model(self):
        model = Sequential()
        model.add(
            Conv2D(
                128,
                21,
                padding='same',
                activation='relu',
                input_shape=(160, 90, 3),
            ))  # 1
        model.add(
            Conv2D(
                128,
                15,
                padding='same',
                activation='relu',
                input_shape=(160, 90, 128),
            ))  # 2
        model.add(
            Conv2D(
                64,
                15,
                padding='same',
                activation='relu',
                input_shape=(160, 90, 128),
            ))  # 3
        model.add(
            Conv2D(
                64,
                9,
                padding='same',
                activation='relu',
                input_shape=(160, 90, 64),
            ))  # 4
        model.add(
            Conv2D(
                32,
                9,
                padding='same',
                activation='relu',
                input_shape=(160, 90, 64),
            ))  # 5
        model.add(
            Conv2D(
                32,
                9,
                padding='same',
                activation='relu',
                input_shape=(160, 90, 32),
            ))  # 6
        model.add(Dropout(0.15))
        model.add(UpSampling2D())
        model.add(
            Conv2D(
                32,
                3,
                padding='same',
                activation='relu',
                input_shape=(320, 180, 32),
            ))  # 7
        model.add(
            Conv2D(
                32,
                3,
                padding='same',
                activation='relu',
                input_shape=(320, 180, 32),
            ))  # 8
        model.add(
            Conv2D(
                32,
                3,
                padding='same',
                activation='relu',
                input_shape=(320, 180, 32),
            ))  # 9
        model.add(
            Conv2D(
                16,
                1,
                padding='same',
                activation='relu',
                input_shape=(320, 180, 32),
            ))  # 10
        model.add(
            Conv2D(
                16,
                1,
                padding='same',
                activation='relu',
                input_shape=(320, 180, 16),
            ))  # 11
        model.add(
            Conv2D(
                16,
                1,
                padding='same',
                activation='relu',
                input_shape=(320, 180, 16),
            ))  # 12
        model.add(
            Conv2D(
                9,
                3,
                padding='same',
                activation='relu',
                input_shape=(320, 180, 16),
            ))  # 13
        model.add(Dropout(0.15))
        model.add(UpSampling2D())
        model.add(
            Conv2D(
                9,
                5,
                padding='same',
                activation='relu',
                input_shape=(640, 360, 9),
            ))  # 14
        model.add(
            Conv2D(
                3,
                5,
                padding='same',
                activation='relu',
                input_shape=(640, 360, 9),
            ))  # 15
        model.add(Dropout(0.3))
        model.compile(
            loss='mse',
            optimizer='adam',
            metrics=['mse', psnr_loss],
        )
        return model