from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Average, Conv2D, Input, UpSampling2D

from .base import Base
from .loss_func import psnr_loss


class ConvolutionAvg(Base):
    def _model(self):
        inp = Input(shape=(160, 90, 3))
        layer = Conv2D(
            64,
            9,
            padding='same',
            activation='relu',
            input_shape=(160, 90, 3),
        )(inp)
        layer = UpSampling2D()(layer)
        layer1 = Conv2D(
            32,
            1,
            padding='same',
            activation='relu',
            input_shape=(320, 180, 64),
        )(layer)
        layer2 = Conv2D(
            32,
            3,
            padding='same',
            activation='relu',
            input_shape=(320, 180, 64),
        )(layer)
        layer3 = Conv2D(
            32,
            5,
            padding='same',
            activation='relu',
            input_shape=(320, 180, 64),
        )(layer)
        layer = Average()([layer1, layer2, layer3])
        layer = UpSampling2D()(layer)
        layer1 = Conv2D(
            3,
            1,
            padding='same',
            activation='relu',
            input_shape=(640, 360, 32),
        )(layer)
        layer2 = Conv2D(
            3,
            3,
            padding='same',
            activation='relu',
            input_shape=(640, 360, 32),
        )(layer)
        layer3 = Conv2D(
            3,
            3,
            padding='same',
            activation='relu',
            input_shape=(640, 360, 32),
        )(layer)
        layer = Average()([layer1, layer2, layer3])
        out = Conv2D(
            3,
            5,
            padding='same',
            activation='relu',
            input_shape=(640, 360, 32),
        )(layer)
        model = Model(inp, out)
        model.compile(
            optimizer=Adam(lr=1e-3),
            loss='mse',
            metrics=[psnr_loss],
        )
        return model