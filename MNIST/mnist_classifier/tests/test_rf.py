import unittest
from MNIST.mnist_classifier import random_forest
from mnist_classifier import utils

class TestModel(unittest.TestCase):
    def test_model(self):
        X_train, y_train, X_test, y_test = utils.load_mnist()
        model = random_forest.create_model()
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        self.assertGreater(accuracy, 0.95)

if __name__ == '__main__':
    unittest.main()