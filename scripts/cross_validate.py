import os
import time
import importlib
import argparse
import pandas as pd

from utils.data import clean_training_samples, resample_classes, DataGenerator
from sklearn.model_selection import StratifiedKFold


def load_model(model_path):
    model = model_path.replace('.py', '').replace('/', '.')
    model = importlib.import_module(model)
    return model.model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('model_path',               type=str, help='model filepath')
    parser.add_argument('--epochs',     default=10, type=int, help='number of epochs per fold',   required=False)
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size for training',     required=False)
    parser.add_argument('--cv_folds',   default=3,  type=int, help='cross validation folds',      required=False)
    parser.add_argument('--workers',    default=1,  type=int, help='number of concurent workers', required=False)
    args = parser.parse_args()

    model = load_model(args.model_path)

    samples = pd.read_csv('data/raw/train.csv')
    samples = clean_training_samples(samples, 'data/raw/train_images/')

    kfolds = StratifiedKFold(n_splits=args.cv_folds, shuffle=True).split(samples.image_id, samples.has_defect)

    for index, (train, validation) in enumerate(kfolds):
        print('Training on fold {}/{}'.format(index + 1, args.cv_folds))
        train      = resample_classes(samples.iloc[train], [None, None, None, None, None])
        validation = samples.iloc[validation]

        train_generator =      DataGenerator(train,      model.input.shape[1:], batch_size=args.batch_size, augmentations=True)
        validation_generator = DataGenerator(validation, model.input.shape[1:], batch_size=args.batch_size, augmentations=False)

        history = model.fit_generator(
            generator=train_generator,
            validation_data=validation_generator,
            use_multiprocessing=True,
            workers=args.workers,
            epochs=args.epochs,
            callbacks=[]
        )
