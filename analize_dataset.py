import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def plot_csv(path):
    # Define the path
    plot_path = 'plot/'
    os.makedirs(plot_path, exist_ok=True)

    # Load the data
    data = pd.read_csv(path)
    # Strip all the spaces in the column names
    data.columns = data.columns.str.strip()
    # Strip all the spaces in the "Common Name" column
    data['Common Name'] = data['Common Name'].str.strip()

    # Create a new column with the total number of images
    data['Total Images'] = data['Train Images'] + data['Validation Images']

    # Plot "Common Name" and "Train Images" count and "Validation Images" on top of it
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.barh(data['Common Name'], data['Train Images'], color='blue')
    ax.barh(data['Common Name'], data['Validation Images'], left=data['Train Images'], color='orange')
    ax.set_xlabel('Number of Images')
    ax.set_ylabel('Common Name')
    ax.set_title('Number of Images per Class')
    ax.legend(['Train Images', 'Validation Images'])
    plt.tight_layout()
    plt.savefig(plot_path + 'monkey_labels.png', dpi=600)


def main():
    # Plot the accuracy
    plot_csv('data/monkey_labels.txt')
    

if "__main__" == __name__:
    main()