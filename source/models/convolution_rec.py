from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ConvLSTM2D, Reshape, UpSampling2D

from .base import Base
from .loss_func import psnr_loss


class ConvolutionRec(Base):
    def _model(self):
        model = Sequential()
        model.add(
            Conv2D(
                64,
                9,
                padding='same',
                activation='relu',
                input_shape=(160, 90, 3),
            ))
        model.add(UpSampling2D())
        model.add(Reshape(target_shape=(1, 320, 180, 64)))
        model.add(
            ConvLSTM2D(
                64,
                3,
                padding='same',
                activation='relu',
                input_shape=(None, 320, 180, 32),
            ))
        model.add(Reshape(target_shape=(320, 180, 64)))
        model.add(
            Conv2D(
                32,
                1,
                padding='same',
                activation='relu',
                input_shape=(320, 180, 64),
            ))
        model.add(UpSampling2D())
        model.add(Reshape(target_shape=(1, 640, 360, 32)))
        model.add(
            ConvLSTM2D(
                32,
                3,
                padding='same',
                activation='relu',
                input_shape=(None, 640, 360, 32),
            ))
        model.add(Reshape(target_shape=(640, 360, 32)))
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