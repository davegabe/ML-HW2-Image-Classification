import pickle
import tensorflow as tf


class CNN(tf.keras.Model):
    def __init__(self, num_classes: int, input_shape: tuple[int, int, int]):
        super(CNN, self).__init__()
        self.in_shape = input_shape
        self.cnn = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPool2D()
        ], name='cnn')
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ], name='classifier')

    def call(self, x: tf.Tensor) -> tf.Tensor:
        y: tf.Tensor = self.classifier(self.cnn(x)) # type: ignore
        return y

    def model(self) -> tf.keras.Model:
        x: tf.Tensor = tf.keras.Input(shape=self.in_shape)  # type: ignore
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def save(self, path: str, history: tf.keras.callbacks.History):
        self.save_weights(path + 'cnn-model.h5')
        with open(path + 'cnn-history.data', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    def load(self, path: str) -> dict:
        self.model().load_weights(path + 'cnn-model.h5')
        history = dict()
        with open(path + 'cnn-history.data', "rb") as file_pi:
            history = pickle.load(file_pi)
        return history

class AlexNET(tf.keras.Model):
    def __init__(self, num_classes: int, input_shape: tuple[int, int, int]):
        super(AlexNET, self).__init__()
        self.in_shape = input_shape
        self.alexnet_cnn = tf.keras.Sequential([
            tf.keras.layers.Conv2D(96, 11, 4, activation='relu', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            tf.keras.layers.Conv2D(256, 5, 1, activation='relu', padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            tf.keras.layers.Conv2D(384, 3, 1, activation='relu', padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(384, 3, 1, activation='relu', padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(256, 3, 1, activation='relu', padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        ], name='alexnet_cnn')
        self.alexnet_classifier = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ], name='alexnet_classifier')

    def call(self, x: tf.Tensor) -> tf.Tensor:
        y: tf.Tensor = self.alexnet_classifier(self.alexnet_cnn(x))  # type: ignore
        return y

    def model(self) -> tf.keras.Model:
        x: tf.Tensor = tf.keras.Input(shape=self.in_shape)  # type: ignore
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def save(self, path: str, history: tf.keras.callbacks.History):
        self.save_weights(path + 'alexnet-model.h5')
        with open(path + 'alexnet-history.data', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    def load(self, path: str) -> dict:
        self.model().load_weights(path + 'alexnet-model.h5')
        history = dict()
        with open(path + 'alexnet-history.data', "rb") as file_pi:
            history = pickle.load(file_pi)
        return history