import unittest

import mnist


class TestMNIST(unittest.TestCase):

    def setUp(self):
        self.dataset = mnist.MNIST()

    def test_num_train_samples(self):
        self.assertEqual(self.dataset.num_train_samples, 60000)

    def test_num_test_samples(self):
        self.assertEqual(self.dataset.num_test_samples, 10000)
