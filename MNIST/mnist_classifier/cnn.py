from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from mnist_classifier.utils import download_mnist

def create_model():
    X_train, y_train, X_test, y_test = download_mnist()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=128)

    return model