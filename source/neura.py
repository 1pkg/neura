import time
from argparse import ArgumentParser

argparser = ArgumentParser(description='neura')

argparser.add_argument(
    '--model_names',
    nargs='+',
    type=str,
    help='names of model for working with',
)

argparser.add_argument(
    '--model_refresh',
    type=bool,
    default=False,
    help='force model train, even if model footprint was found',
)

argparser.add_argument(
    '--bundle',
    type=str,
    help='name of bundle for model training and testing',
)

argparser.add_argument(
    '--bundle_size',
    type=int,
    default=32,
    help='size of bundles for model training and testing',
)

argparser.add_argument(
    '--bundle_epochs',
    type=int,
    default=8,
    help='number of epochs for model training',
)

argparser.add_argument(
    '--bundle_steps_per_epoch',
    type=int,
    default=128,
    help='number of steps per epoch for model training',
)

argparser.add_argument(
    '--run_corruptor',
    type=bool,
    default=False,
    help='run corruptor before all actions',
)

argparser.add_argument(
    '--corruptor_image_quality',
    type=int,
    default=25,
    help='result image quality of corruptor script in %',
)

argparser.add_argument(
    '--corruptor_image_blur',
    type=int,
    default=1,
    help='result image blur of corruptor script in px',
)

argparser.add_argument(
    '--corruptor_image_size_factor',
    type=int,
    default=4,
    help='result image size factor downgrade of corruptor script',
)

argparser.add_argument(
    '--corruptor_crop_sorce',
    type=bool,
    default=False,
    help='crop source bundle to common size for corruptor script',
)

argparser.add_argument(
    '--show_train_plots',
    type=bool,
    default=False,
    help='show plots for models training',
)

argparser.add_argument(
    '--verbose',
    type=bool,
    default=True,
    help='verbosity for all actions',
)

arguments = argparser.parse_args()

import utils

if arguments.run_corruptor:

    if arguments.bundle is not None:
        corruptor = utils.Corruptor(
            arguments.bundle,
            arguments.corruptor_image_quality,
            arguments.corruptor_image_blur,
            arguments.corruptor_image_size_factor,
            arguments.corruptor_crop_sorce,
            verbose=arguments.verbose,
        )
        corruptor.run_walk()

import models

rmodels = []
for model_name in arguments.model_names:
    for name, model in models.__dict__.items():
        if name == model_name:
            rmodels.append(model())
            break

for model in rmodels:
    if arguments.model_refresh:
        model.train(
            arguments.bundle,
            arguments.bundle_size,
            arguments.bundle_steps_per_epoch,
            arguments.bundle_epochs,
            arguments.verbose,
        )
    else:
        model.unserialize()
    model.serialize()

for model in rmodels:
    time_start = time.time()
    model.test(
        arguments.bundle,
        arguments.bundle_size,
        arguments.verbose,
    )
    score = model.score(
        arguments.bundle,
        arguments.bundle_size,
        arguments.verbose,
    )
    time_diff = time.time() - time_start
    print(model.name(), time_diff, score)
    if arguments.show_train_plots:
        plotter = utils.Plotter(arguments.bundle, model.name())
        plotter.show(model.history())
