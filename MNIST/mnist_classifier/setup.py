from setuptools import setup, find_packages

setup(
    name='mnist_classifier',  # Replace with your desired package name
    version='0.1.0',  # Replace with your desired version number
    description='A package for classifying MNIST images with a Keras Convolutional model',
    author='Kulish Dmytro',
    author_email='kulish.dmytro@gmail.com',
    packages=find_packages(),
    install_requires=[  # List required dependencies (e.g., tensorflow, keras)
        'tensorflow',
        'keras', 
        'scikit-learn'
    ],
)