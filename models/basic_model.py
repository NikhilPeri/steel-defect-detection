from keras.layers import *
from keras.models import *

inputs = Input((192, 956, 3))
model = Model(input=inputs, outputs=inputs.output[:, :, :, 0])
