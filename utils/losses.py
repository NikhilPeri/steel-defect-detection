import os
import numpy as np
import cv2 as cv

from keras import backend as K
import keras

from segmentation_models.metrics import iou_score, dice_score
from segmentation_models.losses import dice_loss, categorical_crossentropy, binary_crossentropy

def tversk_loss(y_true, y_pred, alpha=0.7):
    truepos = K.sum(y_true * y_pred, axis=(1,2))
    falsepos = (1 - alpha) * K.sum(y_pred * (1 - y_true), axis=(1,2))
    falseneg = alpha * K.sum((1 - y_pred) * y_true, axis=(1,2))
    tversky = 1 - ((truepos + K.epsilon()) / (truepos +  falsepos + falseneg + K.epsilon()))
    return tversky

def weighted_tversky_loss(weights, alpha=0.7, smooth=1e-10):
    def _tversky_loss(y_true, y_pred):
        truepos = K.sum(y_true * y_pred, axis=(0,1,2))
        falsepos = (1 - alpha) * K.sum(y_pred * (1 - y_true), axis=(0,1,2))
        falseneg = alpha * K.sum((1 - y_pred) * y_true, axis=(0,1,2))
        tversky = 1 - ((truepos + K.epsilon()) / (truepos +  falsepos + falseneg + K.epsilon()))
        return K.pow(tversky, K.constant(weights))

    return _tversky_loss


def dice_bce_loss(y_pred, y_true):
    return dice_loss(y_pred, y_true) + binary_crossentropy(y_pred, y_true)

def custom_dice_score(y_true, y_pred):
    y_true = y_true[:, :, :, 0:4]
    y_pred = y_pred[:, :, :, 0:4]
    numerator = 2 * K.sum(y_true * y_pred, axis=(1,2)) + K.epsilon()
    denominator = K.sum(y_true + y_pred, axis=(1,2)) + K.epsilon()
    return K.mean(numerator / denominator, axis=-1)

def custom_dice_loss(y_true, y_pred):
    return 1 - custom_dice_score(y_true, y_pred)

def custome_dice_bce(y_true, y_pred):
    dice = K.mean(custom_dice_loss(y_true, y_pred))
    y_true = y_true[:, :, :, 0:4]
    y_pred = y_pred[:, :, :, 0:4]
    return  dice + binary_crossentropy(y_true, y_pred)

def lazy_class_loss(y_true, y_pred):
    y_pred = K.sum(K.pow(y_pred, 10), axis=(1, 2))
    y_pred = K.sigmoid(K.constant(10.)*y_pred-K.constant(5.))

    y_true = K.sum(K.pow(y_true, 10), axis=(1, 2))
    y_true = K.sigmoid(K.constant(10.)*y_true - K.constant(5.))

    return K.binary_crossentropy(y_true, y_pred, from_logits=True)

def lazy_class_accuracy(y_true, y_pred):
    y_pred = K.sum(K.sign(y_pred - 0.5), axis=(1, 2))
    y_pred = K.sigmoid(K.constant(10.)*y_pred-K.constant(5.))

    y_true = K.sum(K.sign(y_true - 0.5), axis=(1, 2))
    y_true = K.sigmoid(K.constant(10.)*y_true - K.constant(5.))

    return keras.metrics.binary_accuracy(y_pred, y_true)

def lazy_loss(y_true, y_pred):
    return custom_dice_loss(y_true, y_pred) + lazy_class_loss(y_true, y_pred)

#https://arxiv.org/pdf/1707.03237.pdf
#weights [33., 157., 1., 5.] region_size
def weighted_dice_loss(weights):
    weights = K.variable(np.array(weights))
    def _generalized_dice_loss(y_pred, y_true):
        numerator = 2 * K.sum(y_true * y_pred, axis=(0,1,2))
        denominator = K.sum(y_true + y_pred, axis=(0,1,2)) + K.epsilon()
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
