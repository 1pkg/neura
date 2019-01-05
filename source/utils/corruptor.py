from os import path, listdir, makedirs
from shutil import rmtree
from PIL import Image, ImageFile, ImageFilter

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Corruptor:
    def __init__(
            self,
            bundle,
            image_quality=25,
            image_blur=1,
            image_size_factor=4,
            with_crop=False,
            crop_height=640,
            crop_width=360,
            verbose=True,
    ):
        path_base = path.join(
            path.dirname(__file__),
            '..',
            '..',
            'dump',
        )
        self.__path_train = path.join(path_base, 'train', bundle)
        self.__path_test = path.join(path_base, 'test', bundle)
        self.__verbose = verbose
        self.__image_quality = image_quality
        self.__image_blur = image_blur
        self.__image_size_factor = image_size_factor
        self.__with_crop = with_crop
        self.__crop_height = crop_height
        self.__crop_width = crop_width

    def run_walk(self):
        data_path_src = f'{self.__path_train}_src'
        data_path_mod = f'{self.__path_train}_mod'
        self.__run_walk_internal(data_path_src, data_path_mod)
        data_path_src = f'{self.__path_test}_src'
        data_path_mod = f'{self.__path_test}_mod'
        self.__run_walk_internal(data_path_src, data_path_mod)

    def __run_walk_internal(self, path_src, path_dst):
        if not path.exists(path_src):
            return

        if path.exists(path_dst):
            rmtree(path_dst)
        makedirs(path_dst)

        index = 0
        path_list = listdir(path_src)
        for file_name in path_list:
            file_name_src = path.join(path_src, file_name)
            file_name_dst = path.join(path_dst, file_name)
            if (path.isfile(file_name_src)):
                index += 1
                if self.__with_crop:
                    self.__run_single_crop(file_name_src)
                self.__run_single_internal(file_name_src, file_name_dst)
                if self.__verbose and index % 500 == 0:
                    msg = f'corruptor processing {path_src} | done {index} of {len(path_list)}'
                    print(msg)

    def __run_single_internal(self, path_src, path_dst):
        with Image.open(path_src) as image:
            image = image.convert('RGB')
            if self.__image_size_factor:
                image = image.resize((
                    int(image.size[0] / self.__image_size_factor),
                    int(image.size[1] / self.__image_size_factor),
                ), Image.NEAREST)
            blur = ImageFilter.GaussianBlur(radius=self.__image_blur)
            image = image.filter(blur)
            image.save(
                path_dst,
                'JPEG',
                quality=self.__image_quality,
                optimize=True,
                progressive=True,
            )

    def __run_single_crop(self, path_src):
        with Image.open(path_src) as image:
            image = image.convert('RGB')
            width, height = image.size
            desire_height = self.__crop_height
            desire_width = self.__crop_width

            ratio_width = desire_width / width
            ratio_height = desire_height / height
            if ratio_width > ratio_height:
                new_size = (desire_width, int(height * ratio_width))
            else:
                new_size = (int(width * ratio_height), desire_height)

            image = image.resize(new_size, Image.NEAREST)
            left = int((new_size[0] - desire_width) / 2.0)
            top = int((new_size[1] - desire_height) / 2.0)
            right = int((new_size[0] + desire_width) / 2.0)
            bottom = int((new_size[1] + desire_height) / 2.0)
            image = image.crop((left, top, right, bottom))
            image.save(
                path_src,
                'JPEG',
                optimize=True,
                progressive=True,
            )