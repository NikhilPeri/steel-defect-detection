import keras
import cv2 as cv
import numpy as np
import pandas as pd

from sklearn.utils import resample
from utils.conversion import rle_to_mask

def clean_training_samples(samples, image_dir):
    samples.rename(columns={'EncodedPixels':'encoded_pixels'}, inplace=True)

    split = samples['ImageId_ClassId'].str.split('_', expand=True)
    samples['id'], samples['image_id'], samples['class_id'] = split[0], image_dir + split[0], split[1]
    samples = samples.drop('ImageId_ClassId', axis=1)

    # denormalize class labels
    class_1 = samples[samples.class_id == '1'].drop('class_id', axis=1).rename(columns={'encoded_pixels':'class_1_encoded_pixels'})
    class_2 = samples[samples.class_id == '2'].drop('class_id', axis=1).rename(columns={'encoded_pixels':'class_2_encoded_pixels'})
    class_3 = samples[samples.class_id == '3'].drop('class_id', axis=1).rename(columns={'encoded_pixels':'class_3_encoded_pixels'})
    class_4 = samples[samples.class_id == '4'].drop('class_id', axis=1).rename(columns={'encoded_pixels':'class_4_encoded_pixels'})

    denormalized_train = class_1.merge(class_2, on=['id','image_id']).merge(class_3, on=['id','image_id']).merge(class_4, on=['id','image_id'])
    denormalized_train['has_defect'] = ~(
        denormalized_train['class_1_encoded_pixels'].isna() &
        denormalized_train['class_2_encoded_pixels'].isna() &
        denormalized_train['class_3_encoded_pixels'].isna() &
        denormalized_train['class_4_encoded_pixels'].isna()
    )
    denormalized_train.fillna('', inplace=True)
    denormalized_train['class_1'] = denormalized_train['class_1_encoded_pixels'] != ''
    denormalized_train['class_2'] = denormalized_train['class_2_encoded_pixels'] != ''
    denormalized_train['class_3'] = denormalized_train['class_3_encoded_pixels'] != ''
    denormalized_train['class_4'] = denormalized_train['class_4_encoded_pixels'] != ''
    denormalized_train['class']   = denormalized_train.has_defect.astype(np.uint8)  + denormalized_train[['class_1', 'class_2', 'class_3', 'class_4']].values.astype(np.uint8).argmax(axis=1)

    return denormalized_train

def load_sample(sample, scale=(256, 1600, 3)):
    image = cv.imread(sample.image_id)

    labels = np.dstack([
        cv.resize(rle_to_mask(sample.class_1_encoded_pixels, image), (scale[1], scale[0]), interpolation=cv.INTER_NEAREST),
        cv.resize(rle_to_mask(sample.class_2_encoded_pixels, image), (scale[1], scale[0]), interpolation=cv.INTER_NEAREST),
        cv.resize(rle_to_mask(sample.class_3_encoded_pixels, image), (scale[1], scale[0]), interpolation=cv.INTER_NEAREST),
        cv.resize(rle_to_mask(sample.class_4_encoded_pixels, image), (scale[1], scale[0]), interpolation=cv.INTER_NEAREST),
    ])
    image = cv.resize(image, (scale[1], scale[0])).astype(np.int8)
    if scale[-1] == 1:
        image = np.expand_dims(image[:, :, 0], 2)
    return image, labels

def augment_sample(image, labels):
    if np.random.random() > 0.25:
        flip = np.random.choice([0, 1, -1])
        image = cv.flip(image, flip)
        labels = cv.flip(labels, flip)

    return image, labels

def resample_classes(samples, resampled_classes):
    for c, n_samples in enumerate(resampled_classes):
        resampled_classes[c] = resample(
            samples[samples['class'] == c],
            replace=True,
            n_samples=n_samples,
            random_state=420
        )
    return pd.concat(resampled_classes).reset_index(drop=True)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, samples, scale, batch_size=32, shuffle=True, augmentations=True):
        self.samples = samples
        self.batch_size = batch_size
        self.scale = scale
        self.shuffle = shuffle
        self.augmentations = augmentations

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        samples = self.samples.iloc[index*self.batch_size : (index+1)*self.batch_size]
        images, labels = [], []
        for _, s in samples.iterrows():
            image, label = load_sample(s, scale=self.scale)
            if self.augmentations:
                image, label = augment_sample(image, label)
            images.append(image)
            labels.append(label)

        return np.array(images), np.array(labels)

    def on_epoch_end(self):
        if self.shuffle == True:
            self.samples = self.samples.reindex(np.random.permutation(self.samples.index))
