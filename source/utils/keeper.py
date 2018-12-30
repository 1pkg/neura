from os import path, makedirs
from shutil import rmtree
from PIL import Image
import numpy


class Keeper:
    def __init__(self, bundle, model):
        path_base = path.join(
            path.dirname(__file__),
            '..',
            '..',
            'dump',
        )
        self.__path_result = path.join(path_base, 'result', bundle, model)
        if path.exists(self.__path_result):
            rmtree(self.__path_result)

    def save(self, file_name, data_src, data_mod, data_res):
        path_res = path.join(self.__path_result, file_name)
        if path.exists(path_res):
            rmtree(path_res)
        makedirs(path_res)

        data_src *= 255
        image = Image.fromarray(numpy.uint8(data_src))
        image = image.convert('RGB')
        file_name = path.join(path_res, 'src.jpeg')
        image.save(file_name, 'JPEG')

        data_mod *= 255
        image = Image.fromarray(numpy.uint8(data_mod))
        image = image.convert('RGB')
        file_name = path.join(path_res, 'mod.jpeg')
        image.save(file_name, 'JPEG')

        data_res *= 255
        image = Image.fromarray(numpy.uint8(data_res))
        image = image.convert('RGB')
        file_name = path.join(path_res, 'res.jpeg')
        image.save(file_name, 'JPEG')
