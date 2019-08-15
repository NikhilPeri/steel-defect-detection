import numpy as np

def rle_to_mask(rle_string, image):
    height, width, _ = image.shape
    try:
        mask = np.zeros(height*width, dtype=np.uint8)
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1,2)
        for index, length in rle_pairs:
            index -= 1
            mask[index:index+length] = 1
        mask = mask.reshape(width, height)
        return mask.T
    except:
        return np.zeros((height, width), dtype=np.uint8)

def rle_to_mask(image):
    pixels= image.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
