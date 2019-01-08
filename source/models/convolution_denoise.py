from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Average, Conv2D, Convolution2DTranspose, Input, UpSampling2D

from .base import Base
from .loss_func import psnr_loss


class ConvolutionDenoise(Base):
    def _model(self):
        inp = Input(shape=(160, 90, 3))
        layer = Conv2D(
            64,
            3,
            padding='same',
            activation='relu',
            input_shape=(160, 90, 3),
        )(inp)
        layer = UpSampling2D()(layer)
        layer_conv = Conv2D(
            64,
            3,
            padding='same',
            activation='relu',
            input_shape=(320, 180, 64),
        )(layer)
        layer_deconv = Convolution2DTranspose(
            64,
            3,
            padding='same',
            activation='relu',
            input_shape=(320, 180, 64),
        )(layer_conv)
        layer = Average()([layer_conv, layer_deconv])
        layer = UpSampling2D()(layer)
        layer_conv = Conv2D(
            64,
            3,
            padding='same',
            activation='relu',
            input_shape=(640, 360, 64),
        )(layer)
        layer_deconv = Convolution2DTranspose(
            64,
            3,
            padding='same',
            activation='relu',
            input_shape=(640, 360, 64),
        )(layer)
        layer = Average()([layer_conv, layer_deconv])
        out = Conv2D(
            3,
            5,
            padding='same',
            activation='relu',
            input_shape=(640, 360, 64),
        )(layer)
        model = Model(inp, out)
        model.compile(
            optimizer=Adam(lr=1e-3),
            loss='mse',
            metrics=[psnr_loss],
        )
        return model