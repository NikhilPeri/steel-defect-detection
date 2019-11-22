import os
import sys
import numpy as np
import logging
import cv2 as cv
from utils.data import load_sample, display_sample
from utils.conversion import mask_to_rle

from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from matplotlib import pyplot as plt

def configure_logging(dir):
    formatter = logging.Formatter(
        fmt='[%(levelname)s] %(asctime)s.%(msecs)03d | %(message)s',
        datefmt='%Y-%m-%d %I:%M:%-S'
    )

    file_stream = logging.StreamHandler(stream=open(dir,mode='w+'))
    file_stream.setFormatter(formatter)
    file_stream.setLevel(20)

    console_stream = logging.StreamHandler(stream=sys.stdout)
    console_stream.setFormatter(formatter)
    console_stream.setLevel(30)

    logging.getLogger().setLevel(20)
    logging.getLogger().addHandler(file_stream)
    logging.getLogger().addHandler(console_stream)

def save_checkpoint(dir):
    if not os.path.exists(dir):
        os.makedirs(dir,exist_ok=True)

    return ModelCheckpoint(
        os.path.join(dir, 'best_model.h5'),
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='min'
    )

early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=3,
    mode='min'
)

class VisualizeEpoch(Callback):
    def __init__(self, dir, samples, epoch=1):
        if not os.path.exists(dir):
            os.makedirs(dir,exist_ok=True)
        self.dir = dir
        self.samples = samples
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs):
        if epoch % self.epoch != 0:
            return
        images = [load_sample(i, scale=self.model.input_shape[1:3])[0] for _, i in self.samples.iterrows()]
        predictions = self.model.predict(np.array(images))
        for i in range(len(images)):
            plt.subplot(len(images), 1, i+1)
            display_sample(images[i], predictions[i])
            logging.info('Visualize Epoch {} - Class {} - {}'.format(epoch, self.samples.iloc[i]['class'], self.samples.iloc[i].id.split('.')[0]))

        plt.savefig('{}/epoch_{}'.format(self.dir, epoch))
        plt.close()
