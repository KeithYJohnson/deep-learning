import numpy as np

def softmax(x):
    np_x = np.array(x)
    return np.apply_along_axis(softmax_array_or_column, 0, np_x)

def softmax_array_or_column(x):
    return np.exp(x) / np.sum(np.exp(x))
