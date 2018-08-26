import gzip
import os
import tempfile
from urllib.request import urlretrieve

import numpy as np


def to_one_hot(labels, num_classes=10):
    num_labels = len(labels)
    y = np.zeros(shape=(num_labels, num_classes))
    y[np.arange(num_labels), labels] = 1
    return y


def load(path, offset, shape=None):
    with gzip.open(path) as gz:
        output = np.frombuffer(gz.read(), dtype=np.uint8, offset=offset)
        if shape:
            output = output.reshape(shape)
        return output


class MNIST(object):
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    ]

    def __init__(self, root=None, shape=(-1, 1, 28, 28), one_hot=False):
        self.root = root if root else tempfile.TemporaryDirectory().name
        self.shape = shape
        self.one_hot = one_hot

        self._download()

        train_images_file = os.path.join(self.root,
                                         'train-images-idx3-ubyte.gz')
        train_labels_file = os.path.join(self.root,
                                         'train-labels-idx1-ubyte.gz')
        test_images_file = os.path.join(self.root, 't10k-images-idx3-ubyte.gz')
        test_labels_file = os.path.join(self.root, 't10k-labels-idx1-ubyte.gz')

        self.train_images = load(train_images_file, offset=16, shape=shape)
        self.train_labels = load(train_labels_file, offset=8)
        self.test_images = load(test_images_file, offset=16, shape=shape)
        self.test_labels = load(test_labels_file, offset=8)

        if one_hot:
            self.train_labels = to_one_hot(self.train_labels)
            self.test_labels = to_one_hot(self.test_labels)

    def _download(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        for url in self.urls:
            filename = os.path.join(self.root, url.split('/')[-1])
            if not os.path.exists(filename):
                print('Download {}'.format(url))
                urlretrieve(url, filename=filename)
            else:
                print('{} exists'.format(filename))

    @property
    def num_train_samples(self):
        assert len(self.train_images) == len(self.train_labels)
        return len(self.train_images)

    @property
    def num_test_samples(self):
        assert len(self.test_images) == len(self.test_labels)
        return len(self.test_images)
