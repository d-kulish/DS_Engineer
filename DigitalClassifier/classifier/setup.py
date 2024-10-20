from setuptools import setup, find_packages

setup(
    name="algorithms_mnist",  
    version="0.1.0",  
    description="A package for classifying with MNIST data",
    author="Dmytro Kulish",  
    author_email="kulish.dmytro@gmail.com",  
    packages=find_packages(),  
    install_requires=[  
        "tensorflow",
        "scikit-learn",
        "matplotlib",
        "opencv-python",
    ],
)