import numpy as np
np.random.seed(12345)  # for reproducibility

from os import path
from keras.models import Sequential
from keras.layers import Dropout, Activation
from keras.layers import Conv2D, UpSampling2D
from utils.loader import Loader
from utils.saver import Saver

loader = Loader('anime')
saver = Saver()
model = Sequential()
model.add(
    Conv2D(
        3,
        16,
        padding="same",
        activation='relu',
        input_shape=(160, 90, 3),
    ))
model.add(UpSampling2D())
model.add(Dropout(0.1))
model.add(
    Conv2D(
        3,
        16,
        padding="same",
        activation='relu',
        input_shape=(320, 180, 3),
    ))
model.add(UpSampling2D())
model.add(Dropout(0.25))
model.add(
    Conv2D(
        3,
        8,
        padding="same",
        activation='relu',
        input_shape=(640, 360, 3),
    ))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit_generator(
    loader.gen_data_train(64),
    steps_per_epoch=100,
    epochs=3,
    verbose=1,
)
model.save(path.join(path.dirname(__file__), '..', 'dump', 'convolution.h5'))

score = model.evaluate_generator(
    loader.gen_data_test(64),
    steps=1,
    verbose=1,
)
print(score)

saver.save(model.predict_generator(
    loader.gen_data_test(4),
    steps=1,
))
