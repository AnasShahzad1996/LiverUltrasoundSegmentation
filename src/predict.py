import torch
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.segmentation import felzenszwalb, slic, quickshift


def iou_metric(predict, labels):

    return 0.0

def metric(predict,labels):


    return [], [], []


if __name__=="__main__":
    print ("Reading and predicting the model")


    # Load the image
    image = imread('datasets/raw_data/patient3/2D/Patient-03-ege-010299.png')

    # Apply Felzenszwalb's method
    segments_felzenszwalb = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)

    # Apply SLIC (Simple Linear Iterative Clustering)
    segments_slic = slic(image, n_segments=500, compactness=10, channel_axis=None)

    # Apply Quickshift method
    segments_quickshift = quickshift(np.stack((image, image, image), axis=-1), kernel_size=3, max_dist=6, ratio=0.5)

    # Visualize the superpixel segmentations
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))

    ax[0].imshow(image)
    ax[0].set_title('Original Image')

    ax[1].imshow(segments_felzenszwalb)
    ax[1].set_title('Felzenszwalb')

    ax[2].imshow(segments_slic)
    ax[2].set_title('SLIC')

    ax[3].imshow(segments_quickshift)
    ax[3].set_title('Quickshift')

    for a in ax:
        a.axis('off')

    plt.show()
