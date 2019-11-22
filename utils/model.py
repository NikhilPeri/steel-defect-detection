from segmentation_models import PSPNet
from segmentation_models.backbones import get_feature_layers
import numpy as np
from keras.layers import *
from keras.models import Model
from keras import backend as K

def deep_supervised(backbone_name='', input_shape=(144, 912, 3), encoder_weights='imagenet', activation='sigmoid', classes=4):
    backbone_name = backbone_name[4:]
    model = PSPNet(
        backbone_name=backbone_name,
        input_shape=input_shape,
        encoder_weights=encoder_weights,
        activation=activation,
        classes=classes
    )
    def classification_model(feature_layer):
        feature_layer = GlobalAveragePooling2D()(feature_layer)
        feature_layer = Dropout(0.25)(feature_layer)
        classification = Dense(64, activation='sigmoid', use_bias=False)(feature_layer)
        return Dense(5, activation='sigmoid', use_bias=False)(classification)

    if backbone_name.startswith('seresnet'):
        feature_layer = model.layers[get_feature_layers(backbone_name)[1]].output
    else:
        feature_layer = model.get_layer(get_feature_layers(backbone_name)[1]).output
    classification = classification_model(feature_layer)

    def resize_to_fit(classification):
        classification = ZeroPadding1D((0, input_shape[0] - 5))(classification[..., np.newaxis])[..., np.newaxis]
        classification = K.repeat_elements(classification, 4, axis=-1)
        return classification

    classification = Lambda(resize_to_fit)(classification)

    output = Concatenate(axis=2)([classification, model.output])
    return Model(
        inputs=model.input,
        outputs=output
    )
