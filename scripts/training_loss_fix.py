import glob
import json
import os

from keras.model import load_model
from utils.losses import compute_test_distribution

models = glob.glob('results/model-architecture-*/best_model.h5')
for model_path in models:
    model = load_model(model_path)
    test_distribution = compute_test_distribution(model)
    with open(model_path.replace('best_model.h5', 'training_distribution.json'), 'w+') as fp:
            json.dump(test_distribution, fp, sort_keys=True, indent=4)
