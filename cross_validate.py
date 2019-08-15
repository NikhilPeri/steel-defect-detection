import os
import time
import importlib
import argparse

from utils.data import clean_training_samples, DataGenerator
from sklearn.model_selection import StratifiedKFold


def load_model(model_path):
    model = model_path.replace('.py', '').replace('/', '.')
    model = importlib.import_module(model)

    if 'build_model' not in dir(model):
        raise Exception('Method "build_model" missing {}'.format(model.__file__))

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('model_path',                type=str,   help='model filepath')
    parser.add_argument('--epochs',     default=10,  type=int,   help='number of epochs per fold',      required=False)
    parser.add_argument('--cv_folds',   default=3,   type=int,   help='cross validation folds',         required=False)
    parser.add_argument('--scale',      default=1.0, type=float, help='downsample images for trianing', required=False)
    parser.add_argument('--workers',    default=1,   type=int,   help='number of concurent workers',    required=False)
    args = parser.parse_args()

    model = load_model(args.model_path)
    samples = clean_training_samples()

    kfolds = StratifiedKFold(n_splits=args.cv_folds, shuffle=True).split(samples.image_id, samples.has_defect)

    for index, (train, validation) in enumerate(kfolds):
        print('Training on fold {}/{}'.format(index + 1,args.cv_folds))
        train_generator =      DataGenerator(samples.iloc[train],      scale=args.scale)
        validation_generator = DataGenerator(samples.iloc[validation], scale=args.scale)

        model.build_model(*train_generator.input_output_shape()).fit_generator(
            generator=train_generator,
            validation_data=validation_generator,
            use_multiprocessing=True,
            workers=args.workers,
            epochs=args.epochs
        )
