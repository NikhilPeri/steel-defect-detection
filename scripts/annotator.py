import os
os.chdir('/Users/perinik/github.com/NikhilPeri/steel-defect-detection')
import pandas as pd
import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt
from functools import reduce

from utils.conversion import rle_to_mask, mask_to_rle

CLASSIFICATIONS=[
    ('results/segmentation_submissions/0.88892.csv', 0.88892),
    ('results/segmentation_submissions/0.88902.csv', 0.88902),
    ('results/segmentation_submissions/0.89072.csv', 0.89072)
]
def clean_segmentation(i, df):
    df.columns = ('ImageId_ClassId', 'encoded_pixels_' + str(i))
    df['label_' + str(i)] = ~df['encoded_pixels_' + str(i)].isna()
    return df

def mode_pixels(sample):
    image = cv.imread('data/raw/test_images/'+ sample.ImageId_ClassId.split('_')[0])
    mask_1 = rle_to_mask(sample.encoded_pixels_0, image)
    mask_2 = rle_to_mask(sample.encoded_pixels_1, image)
    mask_3 = rle_to_mask(sample.encoded_pixels_2, image)

    masks = np.concatenate([
        mask_1[..., np.newaxis],
        mask_2[..., np.newaxis],
        mask_3[..., np.newaxis]
    ])
    return mask_to_rle((masks.mean(axis=-1) > (2./3.)).astype(np.int8))

def iou(m1, m2, epsilon=1e-8):
    return (np.sum(m1*m2) + epsilon) / (np.sum(m1 + m2) + epsilon)

def mean_iou(sample):
    image = cv.imread('data/raw/test_images/'+ sample.ImageId_ClassId.split('_')[0])
    mask_1 = rle_to_mask(sample.encoded_pixels_0, image)
    mask_2 = rle_to_mask(sample.encoded_pixels_1, image)
    mask_3 = rle_to_mask(sample.encoded_pixels_2, image)

    mask_1_mask_2 = iou(mask_1, mask_2)
    mask_1_mask_3 = iou(mask_1, mask_3)
    mask_2_mask_3 = iou(mask_2, mask_3)

    return np.mean([mask_1_mask_2, mask_1_mask_3, mask_2_mask_3])

def get_mask(pixels, image):
    return np.concatenate([
        np.zeros(shape=(image.shape[0], image.shape[1], 1)),
        pixels[..., np.newaxis],
        np.zeros(shape=(image.shape[0], image.shape[1], 1)),
        np.zeros(shape=(image.shape[0], image.shape[1], 1)),
    ], axis=-1)

def display_sample(image, labels, fig, i, total):
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

    ax = fig.add_subplot(total, 1, i)
    ax.imshow(image)

def show(sample):
    image = cv.imread('data/raw/test_images/'+ sample.ImageId_ClassId.split('_')[0])
    mask_1 = rle_to_mask(sample.encoded_pixels_0, image)
    mask_2 = rle_to_mask(sample.encoded_pixels_1, image)
    mask_3 = rle_to_mask(sample.encoded_pixels_2, image)
    #mm = rle_to_mask(sample.mode_pixels, image)
    fig = plt.figure()
    display_sample(image, get_mask(mask_1, image), fig, 1, 3)
    display_sample(image, get_mask(mask_2, image), fig, 2, 3)
    display_sample(image, get_mask(mask_3, image), fig, 3, 3)


if __name__ == '__main__':
    results = [clean_segmentation(i, pd.read_csv(r[0])) for i, r in enumerate(CLASSIFICATIONS)]
    res = reduce(lambda l,r: pd.merge(l, r, on=['ImageId_ClassId'], how='inner'), results)
    res.fillna('')

    #res['mean_iou'] = res.apply(mean_iou, axis=1)
    #res['mode_pixels'] = res.apply(mode_pixels, axis=1)
    res['label'] = res['label_0'] | res['label_1'] | res['label_2']
    positives = res[res['label']]
    # 0 - none
    # 1 - mask_1
    # 2 - mask_2
    # 3 - mask_3
    # 4 - mode
    # 5 - false positive
    classes=[]
    plt.ion()
    try:
        for _, positive in positives.iterrows():
            show(positive)
            plt.show()
            x = input('cls')
            plt.close()
            classes.append((positive.ImageId_ClassId, x))
            print("{},{}".format(positive.ImageId_ClassId, x))
    except Exception as e:
        import pdb; pdb.set_trace()
