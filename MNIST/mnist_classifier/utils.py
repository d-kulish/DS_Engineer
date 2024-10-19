from keras.datasets import mnist 
# import tensorflow as tf 

def download_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    return X_train, y_train, X_test, y_test