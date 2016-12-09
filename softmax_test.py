import pytest
import numpy as np
from softmax import *

vector_scores = [1.0, 2.0, 3.0]
expected_vector_softmax = np.array([ 0.09003057, 0.24472847, 0.66524096])

matrix_scores = np.array([
    [1, 2, 3, 6],
    [2, 4, 5, 6],
    [3, 8, 7, 6]
])

expected_matrix_softmax = np.matrix([
 [ 0.09003057,  0.00242826,  0.01587624,  0.33333333],
 [ 0.24472847,  0.01794253,  0.11731043,  0.33333333],
 [ 0.66524096,  0.97962921,  0.86681333,  0.33333333]
])

def test_softmax_vector():
    actual_vector_softmax = softmax(vector_scores)
    assert(np.allclose(actual_vector_softmax, expected_vector_softmax))

def test_softmax_matrix():
    actual_matrix_softmax = softmax(matrix_scores)
    assert(np.allclose(actual_matrix_softmax, expected_matrix_softmax))
