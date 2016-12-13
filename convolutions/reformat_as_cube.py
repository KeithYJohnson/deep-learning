import numpy as np
from params import *

def reformat_as_cube(dataset, labels):
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)
    ).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
