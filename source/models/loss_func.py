from keras import backend as kb
import numpy


def psnr_loss(y_true, y_pred):
    return -10. * kb.log(kb.mean(kb.square(y_pred - y_true)))


def psnr_np(y_true, y_pred):
    return -10. * numpy.log10(numpy.mean(numpy.square(y_pred - y_true)))


def mse_np(y_true, y_pred):
    return numpy.mean(numpy.square(y_pred - y_true))