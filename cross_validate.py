import importutils
from utils.data import clean_training_samples, DataGenerator

if __name__ == '__main__':
    generator = DataGenerator(clean_training_samples())
    generator.__getitem__(1)
    import pdb; pdb.set_trace()
