from keras.models import Sequential
from keras.layers import Conv2D, Dropout, ZeroPadding2D

from .base import Base


class ConvolutionShallowHiepad(Base):
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
        model.add(ZeroPadding2D(padding=(160, 90)))
        model.add(
            Conv2D(
                32,
                1,
                padding='same',
                activation='relu',
                input_shape=(320, 180, 64),
            ))
        model.add(ZeroPadding2D(padding=(320, 180)))
        model.add(
            Conv2D(
                3,
                5,
                padding='same',
                activation='relu',
                input_shape=(640, 360, 32),
            ))
        model.add(Dropout(0.25))
        model.compile(
            loss='mse',
            optimizer='adam',
            metrics=['binary_accuracy'],
        )
        return model