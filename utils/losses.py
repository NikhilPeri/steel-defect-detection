import os
import numpy as np
import cv2 as cv

from keras import backend as K

from segmentation_models.metrics import iou_score, dice_score
from segmentation_models.losses import dice_loss, categorical_crossentropy, binary_crossentropy

def dice_bce_loss(y_pred, y_true):
    return dice_loss(y_pred, y_true) + binary_crossentropy(y_pred, y_true)

#https://arxiv.org/pdf/1707.03237.pdf
#weights [33., 157., 1., 5.] region_size
def weighted_dice_loss(weights):
    weights = K.variable(np.array(weights))
    def _generalized_dice_loss(y_pred, y_true):
        numerator = 2 * K.sum(y_true * y_pred, axis=(0,1,2))
        denominator = K.sum(y_true + y_pred, axis=(0,1,2))
        return K.mean(weights*(1 - numerator / denominator))
    return _generalized_dice_loss


def compute_test_distribution(model):
    test_distribution = {
        'total':   0,
        'class_1': 0,
        'class_2': 0,
        'class_3': 0,
        'class_4': 0,
    }

    for image_id in os.listdir('data/raw/test_images'):
        try:
            image = cv.imread('data/raw/test_images/' + image_id)
            image = cv.resize(image, (model.input_shape[2], model.input_shape[1]))

            mask = model.predict(np.array([image]))[0]
            mask = np.greater(mask, 0.5).astype(np.int8)
            mask = mask.sum(axis=(0, 1))

            test_distribution['total'] += 1
            test_distribution['class_1'] += 1 if mask[0] > 0 else 0
            test_distribution['class_2'] += 1 if mask[1] > 0 else 0
            test_distribution['class_3'] += 1 if mask[2] > 0 else 0
            test_distribution['class_4'] += 1 if mask[3] > 0 else 0
        except:
            pass
    return test_distribution
