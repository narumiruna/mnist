import gzip
import os

import numpy as np

from six.moves.urllib.request import urlretrieve


def to_one_hot(numbers, num_classes=10):
    num_samples = len(numbers)
    y = np.zeros(shape=(num_samples, num_classes))
    y[np.arange(num_samples), numbers] = 1
    return y


def load(path, offset, shape=None):
    with gzip.open(path) as gz:
        output = np.frombuffer(gz.read(), dtype=np.uint8, offset=offset)
        if shape:
            output = output.reshape(shape)
        return output


class MNIST(object):
    urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']

    train_images_file = 'train-images-idx3-ubyte.gz'
    train_labels_file = 'train-labels-idx1-ubyte.gz'
    test_images_file = 't10k-images-idx3-ubyte.gz'
    test_labels_file = 't10k-labels-idx1-ubyte.gz'

    def __init__(self, root, shape=(-1, 1, 28, 28), one_hot=False):
        self.root = root
        self.shape = shape
        self.one_hot = one_hot

        if not self._check_exists():
            print('Dataset not exists.')
            self._download()

        self.train_images = load(os.path.join(self.root, self.train_images_file), offset=16, shape=shape)
        self.train_labels = load(os.path.join(self.root, self.train_labels_file), offset=8)
        self.test_images = load(os.path.join(self.root, self.test_images_file), offset=16, shape=shape)
        self.test_labels = load(os.path.join(self.root, self.test_labels_file), offset=8)

        if one_hot:
            self.train_labels = to_one_hot(self.train_labels)
            self.test_labels = to_one_hot(self.test_labels)

    def _download(self):
        os.makedirs(self.root, exist_ok=True)
        for url in self.urls:
            print('Downloading {}'.format(url))
            filename = os.path.join(self.root, url.split('/')[-1])
            if not os.path.exists(filename):
                urlretrieve(url, filename=filename)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.train_images_file)) and \
            os.path.exists(os.path.join(self.root, self.train_labels_file)) and \
            os.path.exists(os.path.join(self.root, self.test_images_file)) and \
            os.path.exists(os.path.join(self.root, self.test_labels_file))


def test():
    mnist = MNIST('data', one_hot=True)
    print(mnist.train_images.shape)
    print(mnist.train_labels.shape)
    print(mnist.test_images.shape)
    print(mnist.test_labels.shape)


if __name__ == '__main__':
    test()
