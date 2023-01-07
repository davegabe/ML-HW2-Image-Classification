import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

def plot_csv(data):
    # Define the path
    plot_path = 'plot/'
    os.makedirs(plot_path, exist_ok=True)

    # Plot "Common Name" and "Train Images" count and "Validation Images" on top of it
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.barh(data['Common Name'], data['Train Images'], color='blue')
    ax.barh(data['Common Name'], data['Validation Images'], left=data['Train Images'], color='orange')
    ax.set_xlabel('Number of Images')
    ax.set_ylabel('Common Name')
    ax.set_title('Number of Images per Class')
    ax.legend(['Train Images', 'Validation Images'], loc='upper right')
    plt.tight_layout()
    plt.savefig(plot_path + 'monkey_labels.png')
    plt.close()

def plot_grid(train_path, data):
    # Define the path
    plot_path = 'plot/'
    os.makedirs(plot_path, exist_ok=True)

    # Create a new figure with a 5x2 grid
    plt.figure(figsize=(10, 20))
    # For each class, plot one image
    image = None
    for i in range(0, 10):
        # Load a random image from the subfolder n{i}
        path = train_path + f'/n{i}/'
        image = plt.imread(path + np.random.choice(os.listdir(path)))
        # Crop the image to a square using center crop
        h, w, _ = image.shape
        min_dim = min(h, w)
        image = image[(h - min_dim) // 2:(h + min_dim) // 2, (w - min_dim) // 2:(w + min_dim) // 2]
        # Plot the image
        plt.subplot(5, 2, i + 1)
        plt.title(data['Common Name'][i])
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(plot_path + 'data_examples.png')
    plt.close()
    plot_augmentation(image)

def plot_augmentation(image):
    # Plot some examples of data augmentation using ImageDataGenerator
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False
    )
    # Create a new figure with a 5x2 grid
    plt.figure(figsize=(10, 20))
    # Apply some transformations to the image
    for i in range(0, 10):
        # Generate a new image
        new_image = datagen.random_transform(image)
        # Plot the image
        plt.subplot(5, 2, i + 1)
        plt.imshow(new_image)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('plot/data_augmentation.png')
    plt.close()

def main():
    # Define the path
    csv = 'data/monkey_labels.txt'
    # Load the data
    data = pd.read_csv(csv)
    # Strip all the spaces in the column names
    data.columns = data.columns.str.strip()
    # Strip all the spaces in the "Common Name" column
    data['Common Name'] = data['Common Name'].str.strip()
    # Replace '-' with ' ' in the "Common Name" column
    data['Common Name'] = data['Common Name'].str.replace('_', ' ')
    # Capitalize the first letter of each word in the "Common Name" column
    data['Common Name'] = data['Common Name'].str.title() + ' (' + data['Label'].str.strip() + ')'

    # Create a new column with the total number of images
    data['Total Images'] = data['Train Images'] + data['Validation Images']

    # Reverse the order of the rows
    data = data.iloc[::-1]

    # Plot the accuracy
    plot_csv(data)
    # Plot a grid with one image per class
    plot_grid('data/report', data)
    

if "__main__" == __name__:
    main()