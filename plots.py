import os
from matplotlib import pyplot as plt
import tensorflow as tf
from main import load_dataset
from seaborn import heatmap

from models import CNN, AlexNET


def main():
    # Define the parameters
    data_path = 'data/'
    plot_path = 'plot/'
    model_path = 'model/'
    target_size = (256, 256, 3)
    epochs = 250
    os.makedirs(plot_path, exist_ok=True)

    # Load the models
    print("Loading the models...")

    # Load the CNN model
    cnn_model = CNN(num_classes=10, input_shape=target_size)
    cnn_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy', 'mae']
    )
    cnn_history: dict[str, list[float]] = dict()
    if os.path.exists(model_path + 'cnn-model.h5'):
        print("Loading the model...")
        cnn_history = cnn_model.load(model_path)
        # Crop the history to the last epochs
        for key in cnn_history.keys():
            cnn_history[key] = cnn_history[key][-epochs:]

    # Load the AlexNET model
    alexnet_model = AlexNET(num_classes=10, input_shape=target_size)
    alexnet_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    alexnet_history: dict[str, list[float]] = dict()
    if os.path.exists(model_path + 'alexnet-model.h5'):
        print("Loading the model...")
        alexnet_history = alexnet_model.load(model_path)
        # Crop the history to the last epochs
        for key in alexnet_history.keys():
            alexnet_history[key] = alexnet_history[key][-epochs:]

    # Plot the accuracy for the CNN model
    plt.plot(cnn_history['accuracy'], label='Accuracy')
    plt.plot(cnn_history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.yticks([i * 0.1 for i in range(11)])
    plt.xticks([i for i in range(0, epochs, 25)])
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig(plot_path + 'cnn-accuracy.png')
    plt.close()

    # Plot the loss for the CNN model
    plt.plot(cnn_history['loss'], label='Loss', color='red')
    plt.plot(cnn_history['val_loss'], label='Validation Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yticks([i * 0.5 for i in range(15)])
    plt.xticks([i for i in range(0, epochs, 25)])
    plt.ylim([0, min(7, max(cnn_history['val_loss']))])
    plt.legend(loc='upper right')
    plt.savefig(plot_path + 'cnn-loss.png')
    plt.close()

    # Plot the accuracy for the AlexNET model
    plt.plot(alexnet_history['accuracy'], label='Accuracy')
    plt.plot(alexnet_history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.yticks([0.1 * i for i in range(11)])
    plt.xticks([i for i in range(0, epochs, 25)])
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig(plot_path + 'alexnet-accuracy.png')
    plt.close()

    # Plot the loss for the AlexNET model
    plt.plot(alexnet_history['loss'], label='Loss', color='red')
    plt.plot(alexnet_history['val_loss'], label='Validation Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yticks([i * 0.5 for i in range(15)])
    plt.xticks([i for i in range(0, epochs, 25)])
    plt.ylim([0, min(7, max(alexnet_history['val_loss']))])
    plt.legend(loc='upper right')
    plt.savefig(plot_path + 'alexnet-loss.png')
    plt.close()

    # Plot the accuracy
    plt.plot(cnn_history['accuracy'], label='CNN Accuracy')
    plt.plot(cnn_history['val_accuracy'], label='CNN Validation Accuracy')
    plt.plot(alexnet_history['accuracy'], label='AlexNET Accuracy')
    plt.plot(alexnet_history['val_accuracy'], label='AlexNET Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.yticks([0.1 * i for i in range(11)])
    plt.xticks([i for i in range(0, epochs, 25)])
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig(plot_path + 'accuracy.png')
    plt.close()

    # Plot the loss
    plt.plot(cnn_history['loss'], label='CNN Loss')
    plt.plot(cnn_history['val_loss'], label='CNN Validation Loss')
    plt.plot(alexnet_history['loss'], label='AlexNET Loss')
    plt.plot(alexnet_history['val_loss'], label='AlexNET Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yticks([i*0.5 for i in range(15)])
    plt.xticks([i for i in range(0, epochs, 25)])
    plt.ylim([0, 7])
    plt.legend(loc='upper right')
    plt.savefig(plot_path + 'loss.png')
    plt.close()

    # Evaluate the models
    print("Evaluating the models...")
    val_images, val_labels = load_dataset(path=data_path, split='validation', target_size=target_size[:2])

    # Confusion matrix for the CNN model
    print("Confusion matrix for the CNN model...")
    predictions = cnn_model.predict(val_images)
    predictions = tf.argmax(predictions, axis=1)
    confusion_matrix = tf.math.confusion_matrix(val_labels, predictions)
    heatmap(confusion_matrix, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(plot_path + 'cnn-confusion-matrix.png')
    plt.close()

    # F1 score, accuracy, precision and recall for the CNN model
    accuracy = tf.keras.metrics.Accuracy()(val_labels, predictions)
    precision = tf.keras.metrics.Precision()(val_labels, predictions)
    recall = tf.keras.metrics.Recall()(val_labels, predictions)
    f1_score = 2 * (precision * recall) / (precision + recall)
    auc = tf.keras.metrics.AUC()(val_labels, predictions)
    print("For the CNN model:")
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 score: {f1_score}, AUC: {auc}")

    # Confusion matrix for the AlexNET model
    print("Confusion matrix for the AlexNET model...")
    predictions = alexnet_model.predict(val_images)
    predictions = tf.argmax(predictions, axis=1)
    confusion_matrix = tf.math.confusion_matrix(val_labels, predictions)
    heatmap(confusion_matrix, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(plot_path + 'alexnet-confusion-matrix.png')
    plt.close()

    # F1 score, accuracy, precision and recall for the AlexNET model
    accuracy = tf.keras.metrics.Accuracy()(val_labels, predictions)
    precision = tf.keras.metrics.Precision()(val_labels, predictions)
    recall = tf.keras.metrics.Recall()(val_labels, predictions)
    f1_score = 2 * (precision * recall) / (precision + recall)
    auc = tf.keras.metrics.AUC()(val_labels, predictions)
    print("For the AlexNET model:")
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 score: {f1_score}, AUC: {auc}")


if __name__ == '__main__':
    main()
