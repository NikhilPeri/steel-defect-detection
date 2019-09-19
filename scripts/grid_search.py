import os
import re
import logging
import json
import traceback
import pandas as pd
import glob

from keras.optimizers import *
from sklearn.model_selection import ParameterGrid, train_test_split

from segmentation_models import *
from utils.losses import *
from utils.callbacks import *
from utils.data import clean_training_samples, DataGenerator

NAME = 'model-architecture'
dice_unity = weighted_dice_loss([1.,1.,1.,1.])
dice_region_size = weighted_dice_loss([33., 157., 1., 5.])
dice_sample_count = weighted_dice_loss([6., 24., 1., 10.])

param_grid = ParameterGrid({
    'model': [PSPNet],
    'preload': [True],
    'resolution': [(144, 912)],
    'backbone': ['seresnet18'],
    'batch_size': [16],
    'optimizer': [SGD],
    'learning_rate': [0.001, 0.01],
    'momentum': [0.9],
    'loss': [dice_bce_loss, dice_unity, dice_region_size, dice_sample_count],
})
BLACKLIST = [
    lambda p: p['model'] in [Unet, FPN] and p['resolution'] not in [(128, 800), (256, 1600)],
    lambda p: p['model'] not in [Unet, FPN] and p['resolution'] in [(128, 800), (256, 1600)]
]
samples = pd.read_csv('data/raw/train.csv')
samples = clean_training_samples(samples, 'data/raw/train_images/')

train, test = train_test_split(samples.index, stratify=samples['class'], random_state=420)

train = samples.iloc[train]
test = samples.iloc[test]
visualize_samples = pd.concat([
    test[test['class'] == 0].head(2),
    test[test['class'] == 1].head(2),
    test[test['class'] == 2].head(2),
    test[test['class'] == 3].head(2),
    test[test['class'] == 4].head(2)
])

configure_logging('results/' + NAME + '.log')

for params in param_grid:
    if any([blacklist(params) for blacklist in BLACKLIST]):
        continue
    dir = 'results/' + NAME + '-' + \
        params['model'].__name__ + '-' + \
        str(params['resolution'][0]) + 'x' + str(params['resolution'][1]) + '-' + \
        params['backbone']  + '-' + \
        'batch' + str(params['batch_size']) + '-' + \
        params['optimizer'].__name__ + '-' + \
        'lr' + str(params['learning_rate']) + '-' + \
        params['loss'].__name__  + '-' + \
        ('preload' if params['preload'] else 'fresh')
    try:
        os.makedirs(dir, exist_ok=False)
        logging.info(dir)
        model = params['model'](
            backbone_name=params['backbone'],
            input_shape=(*params['resolution'], 3),
            encoder_weights='imagenet',
            activation='sigmoid',
            classes=4,
        )
        if params['preload']:
            pre_trained = glob.glob('../results/*{}-{}-{}*/best_model_*'.format(
                params['model'].__name__ ,
                str(params['resolution'][0]) + 'x' + str(params['resolution'][1]),
                params['backbone'])
            )
            scores = [re.findall('best_model_(\d*\.?\d*)', p) for p in pre_trained]
            scores = [ float(s[0]) if len(s) > 0 else 0.0 for s in scores]
            scores = np.array(scores)
            pre_trained = pre_trained[scores.argmax()]
            model.load_weights(pre_trained)

        model.compile(
            params['optimizer'](lr=params['learning_rate'], momentum=params['momentum']),
            loss=params['loss'],
            metrics=['accuracy', iou_score, dice_score, binary_crossentropy]
        )
        train_generator = DataGenerator(train, model.input.shape[1:], batch_size=params['batch_size'], augmentations=True)
        test_generator =  DataGenerator(test,  model.input.shape[1:], batch_size=params['batch_size'], augmentations=False)

        visualize_epoch =   VisualizeEpoch(dir, visualize_samples, epoch=3)
        history = model.fit_generator(
            generator=train_generator,
            validation_data=test_generator,
            use_multiprocessing=True,
            workers=4,
            verbose=1,
            epochs=50,
            callbacks=[save_checkpoint(dir), early_stopping, visualize_epoch]
        )

        history = pd.DataFrame(history.history)
        history.index.name='epoch'
        history = history.assign(dir=dir)
        history.to_csv(os.path.join(dir, 'training_history.csv'))

        logging.info('compute_test_distribution')
        test_distribution = compute_test_distribution(model)
        logging.info(test_distribution)
        with open(os.path.join(dir, 'test_distribution.json'), 'w') as fp:
            json.dump(test_distribution, fp, sort_keys=True, indent=4)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        logging.error(e)
