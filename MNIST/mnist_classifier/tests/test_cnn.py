import unittest
from MNIST.mnist_classifier import cnn
from mnist_classifier import utils

class TestModel(unittest.TestCase):
    def test_model(self):
        X_train, y_train, _, _ = utils.load_mnist()
        model = cnn.create_model()
        model.fit(X_train, y_train, epochs=1, batch_size=128, verbose=0)
        score = model.evaluate(X_train, y_train, verbose=0)
        self.assertGreater(score[1], 0.95)  # Adjust threshold as needed

if __name__ == '__main__':
    unittest.main()
