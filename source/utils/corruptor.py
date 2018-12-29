from os import path, listdir, makedirs, rmdir
from PIL import Image, ImageFile, ImageFilter

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Corruptor:
    RESULT_IMAGE_QUALITY = 5
    RESULT_IMAGE_BLUR = 3
    RESULT_IMAGE_SIZE_FACTOR = 4

    @classmethod
    def run_walk(cls, path_src, path_dst, verbose=True):
        if path.exists(path_dst):
            rmdir(path_dst)
        makedirs(path_dst)

        index = 0
        path_list = listdir(path_src)
        for file_name in path_list:
            file_name_src = path.join(path_src, file_name)
            file_name_dst = path.join(path_dst, file_name)
            if (path.isfile(file_name_src)):
                index += 1
                cls.run_single(file_name_src, file_name_dst)
                if verbose and index % 500 == 0:
                    print(
                        f'corruptor processing {path_src} | done {index} of {len(path_list)}'
                    )

    @classmethod
    def run_single(cls, path_src, path_dst):
        with Image.open(path_src) as image:
            image = image.convert('RGB')
            image.size
            image = image.resize((
                int(image.size[0] / cls.RESULT_IMAGE_SIZE_FACTOR),
                int(image.size[1] / cls.RESULT_IMAGE_SIZE_FACTOR),
            ))
            blur = ImageFilter.GaussianBlur(radius=cls.RESULT_IMAGE_BLUR)
            image = image.filter(blur)
            image.save(
                path_dst,
                'JPEG',
                quality=cls.RESULT_IMAGE_QUALITY,
                optimize=True,
                progressive=True,
            )