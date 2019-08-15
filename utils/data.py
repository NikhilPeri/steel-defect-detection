import os
import keras

import cv2 as cv
import numpy as np
import pandas as pd

def clean_training_samples():
    train = pd.read_csv('data/raw/train.csv')
    train.rename(columns={'EncodedPixels':'encoded_pixels'}, inplace=True)

    split = train['ImageId_ClassId'].str.split('_', expand=True)
    train['image_id'], train['class_id'] = 'data/raw/train_images/' + split[0], split[1]
    train = train.drop('ImageId_ClassId', axis=1)

    # denormalize class labels
    class_1 = train[train.class_id == '1'].drop('class_id', axis=1).rename(columns={'encoded_pixels':'class_1_encoded_pixels'})
    class_2 = train[train.class_id == '2'].drop('class_id', axis=1).rename(columns={'encoded_pixels':'class_2_encoded_pixels'})
    class_3 = train[train.class_id == '3'].drop('class_id', axis=1).rename(columns={'encoded_pixels':'class_3_encoded_pixels'})
    class_4 = train[train.class_id == '4'].drop('class_id', axis=1).rename(columns={'encoded_pixels':'class_4_encoded_pixels'})

    denormalized_train = class_1.merge(class_2, on='image_id').merge(class_3, on='image_id').merge(class_4, on='image_id')
    denormalized_train['has_defect'] = ~(
        denormalized_train['class_1_encoded_pixels'].isna() &
        denormalized_train['class_2_encoded_pixels'].isna() &
        denormalized_train['class_3_encoded_pixels'].isna() &
        denormalized_train['class_4_encoded_pixels'].isna()
    )
    denormalized_train.fillna('', inplace=True)

    return denormalized_train

def load_sample(sample, downscale=1.0):
    image = cv.imread(os.path.join(self.image_dir, sample.image_id))
    image = cv2.resize(image,None,fx=downscale,fy=downscale)

    labels = np.dstack([
        rle_to_mask(sample.class_1_encoded_pixels, image),
        rle_to_mask(sample.class_2_encoded_pixels, image),
        rle_to_mask(sample.class_3_encoded_pixels, image),
        rle_to_mask(sample.class_4_encoded_pixels, image),
    ])
    defects = labels.sum(axis=2)
    non_defects = np.ones(defects.shape, dtype=defects.dtype) - defects

    return image, np.dstack([labels, non_defects])

class DataGenerator(keras.utils.Sequence):
    def __init__(self, samples, batch_size=32, shuffle=True, downscale=1.0):
        self.samples = samples
        self.batch_size = batch_size
        self.downscale = downscale
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.samples) / self.batch_size))

    def input_output_shape(self):
        input, output = self.__getitem__(0)
        return input.shape[1:], output.shape[1:]

    def __getitem__(self, index):
        samples = self.samples.iloc[index*self.batch_size : (index+1)*self.batch_size]
        images, labels = zip(*[load_sample(s, downscale=self.downscale) for s in samples])

        return images, labels

    def on_epoch_end(self):
        if self.shuffle == True:
            self.samples = self.samples.reindex(np.random.permutation(self.samples.index))
