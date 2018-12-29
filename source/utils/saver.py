from os import path
from PIL import Image
import numpy


class Saver:
    def __init__(self):
        self.__path_base = path.join(
            path.dirname(__file__),
            '..',
            '..',
            'dump',
        )

    def save(self, data):
        index = 0
        for image in data:
            image *= 255
            image = Image.fromarray(numpy.uint8(image))
            index += 1
            file_name = path.join(self.__path_base, str(index) + '.jpeg')
            image.save(file_name, 'JPEG')
