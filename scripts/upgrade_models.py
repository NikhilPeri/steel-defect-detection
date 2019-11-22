import os
import re
import glob
import keras

from segmentation_models import *

MODEL_CLASS = {
    'FPN': FPN,
    'Unet': Unet,
    'PSPNet': PSPNet
}
def upgrade_version(model_path):
    model, resolution, backbone = re.findall('(PSPNet|FPN|Unet)-(\w+)-(\w+)-', model_path)[0]
    resolution = resolution.split('x')
    model = MODEL_CLASS[model](
        backbone_name=backbone,
        input_shape=(144, 816, 3),
        activation='softmax',
        classes=5,
    )
    model.load_weights(model_path)
    model.save(model_path)

if __name__ == '__main__':
    models = glob.glob('results/*/best_model*')
    for m in models:
        print(m)
        upgrade_version(m)
