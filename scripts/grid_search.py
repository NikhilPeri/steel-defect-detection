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
from utils.optimizers import *
from utils.callbacks import *
from utils.model import *
from utils.data import *

NAME = 'crops'

param_grid = ParameterGrid({
    'model': [FPN],
    'preload': [False],
    'resolution': [(128, 128)],
    'backbone': ['seresnet34'],
    'batch_size':[128],
    'optimizer': [Adam],
    'learning_rate': [0.01],
    'momentum':  [0.9],
    'classes': [[0, 1, 2, 3, 4]],
    'loss': [('custome_dice_bce', custome_dice_bce)],
    'generator': [DataGenerator],
    'activation': ['softmax'],
    'crop': [True]
})
BLACKLIST = [
    #lambda p: p['model'] in [Unet, FPN] and p['resolution'] not in [(128, 800), (256, 1600)],
    #lambda p: p['model'] not in [Unet, FPN] and p['resolution'] in [(128, 800), (256, 1600)]
]
samples = pd.read_csv('data/raw/train.csv')
samples = clean_training_samples(samples, 'data/raw/train_images/')

configure_logging('results/' + NAME + '.log')

for params in param_grid:
    if any([blacklist(params) for blacklist in BLACKLIST]):
        continue
    dir = 'results/' + NAME + '-' + \
        params['model'].__name__ + '-' + \
        str(params['resolution'][0]) + 'x' + str(params['resolution'][1]) + '-' + \
        params['backbone']  + '-' + \
        'batch' + str(params['batch_size']) + '-' + \
        params['activation'] + '-' + \
        params['optimizer'].__name__ + '-' + \
        'lr' + str(params['learning_rate']) + '-' + \
        params['loss'][0]  + '-' + \
        'classes[' + ''.join([str(i) for i in params['classes']])  + ']-' + \
        ('preload' if params['preload'] else 'fresh')
    try:
        os.makedirs(dir, exist_ok=True)
        logging.info(dir)
        model = params['model'](
            backbone_name=params['backbone'],
            input_shape=(*params['resolution'], 3),
            encoder_weights='imagenet',
            activation=params['activation'],
            classes=4 if params['activation'] == 'sigmoid' else 5,
        )
        if params['preload']:
            pre_trained = glob.glob('results/crops*{}*{}*{}*/best_model_*'.format(
                params['model'].__name__ ,
                params['backbone'],
                params['activation'])
            )
            scores = [re.findall('best_model_(\d*\.?\d*)', p) for p in pre_trained]
            scores = [ float(s[0]) if len(s) > 0 else 0.0 for s in scores]
            scores = np.array(scores)
            pre_trained = pre_trained[scores.argmax()]
            print('loading: {}'.format(pre_trained))
            model.load_weights(pre_trained)
        model.compile(
            params['optimizer'](lr=params['learning_rate']),# momentum=params['momentum']),
            loss=params['loss'][1],
            metrics=[dice_score, custom_dice_score, custom_dice_loss, binary_crossentropy]
        )
        subset = samples[samples['class'].isin(params['classes'])].reset_index(drop=True)
        '''
        subset = pd.concat([
            samples[samples['class'].isin(params['classes'])],
            clean_training_samples(pd.read_csv('data/hard_class_0.csv')[['ImageId_ClassId', 'EncodedPixels']], 'data/raw/train_images/'),
        ]).reset_index(drop=True)
        '''
        for i in range(1,5):
            if i not in params['classes']:
                subset['class_{}_encoded_pixels'.format(i)] = ''

        train, test = train_test_split(subset.index, stratify=subset['class'], random_state=420)

        train = subset.iloc[train]
        test = subset.iloc[test]

        samples = pd.read_csv('data/raw/train_crops.csv')
        samples = clean_training_samples(samples, 'data/raw/train_crops/')
        samples['id'] = samples['src_image_x']

        if params['crop']:
            train = pd.merge(samples, train[['id']], on='id', how='inner').reset_index(drop=True)
            test =  pd.merge(samples, test[['id']], on='id', how='inner').reset_index(drop=True)

        load_fn = load_sample if params['activation'] =='sigmoid' else load_five_class
        train_generator = params['generator'](train, model.input.shape[1:], batch_size=params['batch_size'], augmentations=True, load_fn=load_fn)
        test_generator =  params['generator'](test,  model.input.shape[1:], batch_size=params['batch_size'], augmentations=False, load_fn=load_fn)

        history = model.fit_generator(
            generator=train_generator,
            validation_data=test_generator,
            use_multiprocessing=True,
            workers=4,
            verbose=1,
            epochs=50,
            callbacks=[save_checkpoint(dir), early_stopping]
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
