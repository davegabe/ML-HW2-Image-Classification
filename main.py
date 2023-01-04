from models import CNN
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def load_dataset(path="data/", split='training', target_size=(32, 32)):
    # Load the dataset
    images = []
    labels = []
    # For each class (n0, n1, ..., n9)
    for i in range(10):
        folder_path = os.path.join(path, split, f'n{i}')
        # For each image in the class
        for file in os.listdir(folder_path):
            # Load the image
            image = tf.keras.preprocessing.image.load_img(os.path.join(folder_path, file), target_size=target_size)
            image = tf.keras.preprocessing.image.img_to_array(image)
            # Normalize the image
            image = image / 255.0
            # Append the image and the label
            images.append(image)
            labels.append(i)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def main():
    # Define the paths
    data_path = 'data/'
    model_path = 'model/'
    plot_path = 'plot/'
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    # Define the parameters
    target_size = (128, 128, 3)
    epochs = 3

    # Load the dataset
    print("Loading the dataset...")
    train_images, train_labels = load_dataset(path=data_path, split='training', target_size=target_size[:2])
    val_images, val_labels = load_dataset(path=data_path, split='validation', target_size=target_size[:2])

    # Augment the dataset
    print("Augmenting the dataset...")
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
    )

    # Train the model
    print("Training the model...")
    model = CNN(num_classes=10, input_shape=target_size)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy', 'mae']
    )

    # Load the model
    old_history: dict = dict()
    if os.path.exists(model_path + 'model.h5'):
        print("Loading the model...")
        # Continue training the model
        epochs -= len(old_history['accuracy'])

    # Train the model
    history: tf.keras.callbacks.History = model.fit(
        train_datagen.flow(train_images, train_labels, batch_size=32),
        epochs=epochs,
        validation_data=(val_images, val_labels)
    )

    # Merge the history
    for key in history.history.keys():
        history.history[key] = old_history.get(key, []) + history.history[key]

    # Plot the accuracy and loss
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(plot_path + 'accuracy.png')

    # Evaluate the model
    print("Evaluating the model...")
    test_acc = model.evaluate(val_images, val_labels)
    print("Test accuracy: ", test_acc)

    # Save the model
    print("Saving the model...")
    model.save(model_path, history)


if __name__ == '__main__':
    main()
