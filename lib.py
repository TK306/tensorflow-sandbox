import tensorflow as tf
import numpy as np
from collections import namedtuple


def mnist2namedtuple(src):
    output = namedtuple("MnistDataset", ["x_train", "y_train", "x_test", "y_test"])
    (output.x_train, output.y_train), (output.x_test, output.y_test) = src
    return output
