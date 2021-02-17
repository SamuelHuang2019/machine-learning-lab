import torch
import torchvision
import matplotlib.pyplot as plt
import os
import struct
import numpy as np


def load_mnist(path, KIND='train'):
    """load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idxl-ubyte' % KIND)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % KIND)
