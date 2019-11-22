import pandas as pd

from sklearn.model_selection import train_test_split

from segmentation_models import *
from utils.losses import *
from utils.optimizers import *
from utils.callbacks import *
from utils.model import *
from utils.data import *

samples = pd.read_csv('data/raw/train.csv')
samples = clean_training_samples(samples, 'data/raw/train_images/')

train, test = train_test_split(subset.index, stratify=subset['class'], random_state=420)
train = samples.iloc[train]
test = samples.iloc[test]

samples = pd.read_csv('data/raw/train_crops.csv')
samples = clean_training_samples(samples, 'data/raw/train_crops/')

train = pd.merge(train, samples, left_on='image_id', right_on='src_image_id', how='inner')
test =  pd.merge(test, samples, left_on='image_id', right_on='src_image_id', how='inner')

model = PSPNet(
    backbone_name='',
    input_shape=(256, 256, 3),
    encoder_weights='imagenet',
    activation='softmax',
    classes=5,
)
model.compile(
    Adam(lr=params['learning_rate']),
    loss=params['loss'][1],
    metrics=[dice_score, custom_dice_score, custom_dice_loss, binary_crossentropy]
)
train_generator = DataGenerator(train, model.input.shape[1:], batch_size=32, augmentations=True, load_fn=load_five_class)
test_generator =  DataGenerator(test,  model.input.shape[1:], batch_size=32, augmentations=False, load_fn=load_five_class)
