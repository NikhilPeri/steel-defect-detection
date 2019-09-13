from keras.applications.densenet import DenseNet121
from keras.layers import *

def build_model(input_shape, output_shape):
    inputs = Input((input_shape))
    downsample = AveragePooling2D(pool_size=(4, 4))(inputs)

    pre_trained = DenseNet121(include_top=False, weights='imagenet', input_tensor=downsample)

    feature_vector = GlobalAveragePooling2D(data_format='channels_last')(pre_trained.output)
    feature_vector = Flatten()(feature_vector)

    fully_connectedA = Dense(64)(feature_vector)
    fully_connectedB = Dense(64)(fully_connectedA)
    output = Dense(output_shape)(concatenate([fully_connectedA, fully_connectedB]))

    model = Model(inputs=inputs, output=output)
    model.summary()

    return model
