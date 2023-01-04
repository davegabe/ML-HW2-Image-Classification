import os
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
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        y: tf.Tensor = self.cnn(x)  # type: ignore
        return y

    def model(self) -> tf.keras.Model:
        x: tf.Tensor = tf.keras.Input(shape=self.in_shape)  # type: ignore
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def save(self, path: str, history: tf.keras.callbacks.History):
        self.save_weights(path + 'model.h5')
        with open(path + 'history.data', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    def load(self, path: str) -> dict:
        self.model().load_weights(path + 'model.h5')
        history = dict()
        with open(path + 'history.data', "rb") as file_pi:
            history = pickle.load(file_pi)
        return history
