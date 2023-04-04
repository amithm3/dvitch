import matplotlib.pyplot as plt

import numpy as np


def show_sample(images, labels, *, rows=4, cols=6, figsize=(5, 5)):
    sample = np.random.choice(images.shape[0], rows * cols, replace=False)
    images = images[sample]
    labels = labels[sample]
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(images[i * cols + j], cmap='gray')
            axes[i, j].set_title(str(labels[i * cols + j]))
            axes[i, j].axis('off')
    plt.show()


def show_image(image, label="", figsize=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap='gray')
    ax.set_title(str(label))
    ax.axis('off')
    plt.show()
