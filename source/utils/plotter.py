from matplotlib import pyplot


class Plotter:
    def __init__(self, bundle, model):
        self.__bundle = bundle
        self.__model = model

    def show(self, history):
        if history is None:
            pass

        pyplot.plot(history['loss'])
        pyplot.title(f'model {self.__model} with bundle {self.__bundle} loss')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.show()

        pyplot.plot(history['mean_squared_error'])
        pyplot.title(f'model {self.__model} with bundle {self.__bundle} mse')
        pyplot.ylabel('mse')
        pyplot.xlabel('epoch')
        pyplot.show()

        pyplot.plot(history['psnr_loss'])
        pyplot.title(f'model {self.__model} with bundle {self.__bundle} psnr')
        pyplot.ylabel('psnr')
        pyplot.xlabel('epoch')
        pyplot.show()
