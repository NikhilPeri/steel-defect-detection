import keras
import cv2 as cv
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.utils import resample
from utils.conversion import rle_to_mask
from skimage import transform

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

    classes = pd.concat([class_1[['id','image_id']], class_2[['id','image_id']], class_3[['id','image_id']], class_4[['id','image_id']]]).drop_duplicates()
    denormalized_train = classes.merge(class_1, on=['id','image_id'], how='left').merge(class_2, on=['id','image_id'], how='left').merge(class_3, on=['id','image_id'], how='left').merge(class_4, on=['id','image_id'], how='left')
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

    return denormalized_train.reset_index(drop=True)

def load_sample(sample, scale=(256, 1600, 3), noise=0.05):
    image = cv.imread(sample.image_id)

    labels = np.dstack([
        cv.resize(rle_to_mask(sample.class_1_encoded_pixels, image), (scale[1], scale[0]), interpolation=cv.INTER_NEAREST),
        cv.resize(rle_to_mask(sample.class_2_encoded_pixels, image), (scale[1], scale[0]), interpolation=cv.INTER_NEAREST),
        cv.resize(rle_to_mask(sample.class_3_encoded_pixels, image), (scale[1], scale[0]), interpolation=cv.INTER_NEAREST),
        cv.resize(rle_to_mask(sample.class_4_encoded_pixels, image), (scale[1], scale[0]), interpolation=cv.INTER_NEAREST),
    ])
    image = cv.resize(image, (scale[1], scale[0]))
    if scale[-1] == 1:
        image = np.expand_dims(image[:, :, 0], 2)
    return image.astype(np.float32), labels.astype(np.float32)

def load_five_class(sample, scale=(256, 1600, 3)):
    image, labels = load_sample(sample, scale=scale)
    background = np.ones(shape=(image.shape[0], image.shape[1], 1))

    background -= np.sum(labels, axis=-1)[..., np.newaxis]
    labels = np.append(labels, background, axis=-1)
    return image, labels

def augment_sample(image, labels):
    rotate = np.random.choice([0, 90, 180, 270])
    image = transform.rotate(image, rotate)
    labels = transform.rotate(labels, rotate)

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

def display_sample(image, labels):
    CLASS_ID_COLOURS = {
        '1': (3,  3, 255),
        '2': (3, 255, 3),
        '3': (255, 3, 3),
        '4': (255, 3, 255),
    }
    LABEL_OPACITY=0.15
    image = image.astype(np.uint8)

    cv.addWeighted(np.multiply(CLASS_ID_COLOURS['1'], np.repeat(np.expand_dims(labels[:, :, 0], axis=2), 3, axis=2)).astype(np.uint8), LABEL_OPACITY, image, 1.0, 0, image)
    cv.addWeighted(np.multiply(CLASS_ID_COLOURS['2'], np.repeat(np.expand_dims(labels[:, :, 1], axis=2), 3, axis=2)).astype(np.uint8), LABEL_OPACITY, image, 1.0, 0, image)
    cv.addWeighted(np.multiply(CLASS_ID_COLOURS['3'], np.repeat(np.expand_dims(labels[:, :, 2], axis=2), 3, axis=2)).astype(np.uint8), LABEL_OPACITY, image, 1.0, 0, image)
    cv.addWeighted(np.multiply(CLASS_ID_COLOURS['4'], np.repeat(np.expand_dims(labels[:, :, 3], axis=2), 3, axis=2)).astype(np.uint8), LABEL_OPACITY, image, 1.0, 0, image)

    plt.imshow(image)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, samples, scale, batch_size=32, shuffle=True, augmentations=True, load_fn=load_sample):
        self.samples = samples
        self.batch_size = batch_size
        self.scale = scale
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.load_fn = load_fn

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        samples = self.samples.iloc[index*self.batch_size : (index+1)*self.batch_size]
        images, labels = [], []
        for _, s in samples.iterrows():
            image, label = self.load_fn(s, scale=self.scale)
            if self.augmentations:
                image, label = augment_sample(image, label)
            images.append(image)
            labels.append(label)

        return np.array(images), np.array(labels)

    def on_epoch_end(self):
        if self.shuffle == True:
            self.samples = self.samples.sample(frac=1).reset_index(drop=True)
