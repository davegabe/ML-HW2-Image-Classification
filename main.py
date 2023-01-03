import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from models import CNN

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
    # Define the dataset
    data_path = 'data/'
    model_path = 'model/'
    target_size = (128, 128, 3)

    # Load the dataset
    print("Loading the dataset...")
    train_images, train_labels = load_dataset(path=data_path, split='training', target_size=target_size[:2])
    val_images, val_labels = load_dataset(path=data_path, split='validation', target_size=target_size[:2])

    # Train the model
    print("Training the model...")
    model = CNN(num_classes=10, input_shape=target_size)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))
    
    # Plot the accuracy and loss
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    # Evaluate the model
    test_loss, test_acc = model.evaluate(val_images,  val_labels)
    print(test_acc)

    # Save the model
    print("Saving the model...")
    model.save(model_path + 'model.h5')


if __name__ == '__main__':
    main()
