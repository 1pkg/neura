from os import path, listdir
from PIL import Image, ImageFile
from random import choice
import numpy

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Loader:
    def __init__(self, bundle, scale_factor=None, scale_type=None):
        self.__path_base = path.join(
            path.dirname(__file__),
            '..',
            '..',
            'dump',
        )
        self.__bundle = bundle
        self.__scale_factor = scale_factor
        self.__scale_type = scale_type

    def gen_data_train(self, batch_size):
        path_train = path.join(self.__path_base, 'train', self.__bundle)
        yield from self.__lazy_data_load(path_train, batch_size)

    def gen_data_test(self, batch_size):
        path_test = path.join(self.__path_base, 'test', self.__bundle)
        yield from self.__lazy_data_load(path_test, batch_size)

    def __lazy_data_load(self, data_path, batch_size):
        data_path_src = f'{data_path}_src'
        data_path_mod = f'{data_path}_mod'
        data_list_src = listdir(data_path_src)
        while True:
            data_batch_src = []
            data_batch_mod = []
            for _ in range(batch_size):
                file_name = choice(data_list_src)
                file_name_src = path.join(data_path_src, file_name)
                file_name_mod = path.join(data_path_mod, file_name)
                if path.isfile(file_name_src) and path.isfile(file_name_mod):
                    with Image.open(file_name_src) as image:
                        data_item_src = numpy.array(image)
                        data_item_src = data_item_src.astype('float32')
                        data_item_src /= 255
                        data_batch_src.append(data_item_src)
                    with Image.open(file_name_mod) as image:
                        if self.__scale_factor:
                            image = image.resize((
                                int(image.size[0] * self.__scale_factor),
                                int(image.size[1] * self.__scale_factor),
                            ), Image.NEAREST)
                        data_item_mod = numpy.array(image)
                        data_item_mod = data_item_mod.astype('float32')
                        data_item_mod /= 255
                        data_batch_mod.append(data_item_mod)
            yield (numpy.array(data_batch_mod), numpy.array(data_batch_src))

    def data_train(self, batch_size):
        path_train = path.join(self.__path_base, 'train', self.__bundle)
        return self.__data_load(path_train, batch_size)

    def data_test(self, batch_size):
        path_train = path.join(self.__path_base, 'test', self.__bundle)
        return self.__data_load(path_train, batch_size)

    def __data_load(self, data_path, batch_size):
        data_path_src = f'{data_path}_src'
        data_path_mod = f'{data_path}_mod'
        data_list_src = listdir(data_path_src)
        data_batch_src = []
        data_batch_mod = []
        for _ in range(batch_size):
            file_name = choice(data_list_src)
            file_name_src = path.join(data_path_src, file_name)
            file_name_mod = path.join(data_path_mod, file_name)
            if path.isfile(file_name_src) and path.isfile(file_name_mod):
                with Image.open(file_name_src) as image:
                    data_item_src = numpy.array(image)
                    data_item_src = data_item_src.astype('float32')
                    data_item_src /= 255
                    data_batch_src.append(data_item_src)
                with Image.open(file_name_mod) as image:
                    if self.__scale_factor and self.__scale_type is not None:
                        image = image.resize((
                            int(image.size[0] * self.__scale_factor),
                            int(image.size[1] * self.__scale_factor),
                        ), self.__scale_type)
                    data_item_mod = numpy.array(image)
                    data_item_mod = data_item_mod.astype('float32')
                    data_item_mod /= 255
                    data_batch_mod.append(data_item_mod)
        return (numpy.array(data_batch_mod), numpy.array(data_batch_src))