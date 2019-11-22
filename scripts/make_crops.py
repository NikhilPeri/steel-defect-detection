import pandas as pd
import numpy as np
from utils.data import clean_training_samples, load_sample
from utils.conversion import *
from matplotlib import pyplot as plt
import cv2 as cv
from tqdm import tqdm

samples = pd.read_csv('data/raw/train.csv')
samples = clean_training_samples(samples, 'data/raw/train_images/')

WIDTH=256
HEIGHT=256

SRC_WIDTH = 1600
SRC_HEIGHT = 256
horiz_slices = np.floor(SRC_WIDTH/WIDTH)
horiz_overlap = np.floor((1-((SRC_WIDTH/WIDTH) % 1))*WIDTH/horiz_slices)

vert_slices = np.floor(SRC_HEIGHT/HEIGHT)
if HEIGHT < 256:
    vert_slices += 1
vert_overlap = np.floor((1-((SRC_HEIGHT/HEIGHT) % 1))*HEIGHT/vert_slices)

def crop_samples(sample):
    image_crops = []
    label_crops = []

    image, label = load_sample(sample)

    # create_crops
    for i in range(int(horiz_slices) + 1):
        horiz_start = int(i*WIDTH - i*(horiz_overlap))
        horiz_stop = int((i + 1)*WIDTH - i*(horiz_overlap))
        if horiz_stop > SRC_WIDTH:
            horiz_start -= (horiz_stop - SRC_WIDTH)
            horiz_stop = SRC_WIDTH

        for j in range(int(vert_slices)):
            vert_start = int(j*HEIGHT - j*(vert_overlap))
            vert_stop = int((j + 1)*HEIGHT - j*(vert_overlap))
            if vert_stop > SRC_HEIGHT:
                vert_start -= (vert_stop - SRC_HEIGHT)
                vert_stop = SRC_HEIGHT

            image_crops.append(image[vert_start:vert_stop, horiz_start:horiz_stop])
            label_crops.append(label[vert_start:vert_stop, horiz_start:horiz_stop])

    return image_crops, label_crops

train_crops = pd.DataFrame(columns=['src_image', 'ImageId_ClassId', 'EncodedPixels'])

for _, sample in tqdm(samples.iterrows()):
    image_crops, label_crops = crop_samples(sample)
    for i in range(len(image_crops)):
        image_id = sample.id.replace('.jpg', str(i) + '.jpg')
        if image_crops[i].mean() < 5:
            continue
        cv.imwrite(
            'data/raw/train_crops/'+ image_id,
            image_crops[i]
        )
        train_crops = train_crops.append({'src_image': sample.id, 'ImageId_ClassId': image_id + '_1', 'EncodedPixels': mask_to_rle(label_crops[i][:, :, 0])}, ignore_index=True)
        train_crops = train_crops.append({'src_image': sample.id, 'ImageId_ClassId': image_id + '_2', 'EncodedPixels': mask_to_rle(label_crops[i][:, :, 1])}, ignore_index=True)
        train_crops = train_crops.append({'src_image': sample.id, 'ImageId_ClassId': image_id + '_3', 'EncodedPixels': mask_to_rle(label_crops[i][:, :, 2])}, ignore_index=True)
        train_crops = train_crops.append({'src_image': sample.id, 'ImageId_ClassId': image_id + '_4', 'EncodedPixels': mask_to_rle(label_crops[i][:, :, 3])}, ignore_index=True)

train_crops.to_csv('data/raw/train_crops.csv', index=False)

import pdb; pdb.set_trace()
