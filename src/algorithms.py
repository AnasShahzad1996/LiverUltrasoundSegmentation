import torch
import requests
import numpy as np
import matplotlib as plt

from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from segment_anything import SamPredictor
from transformers import SamModel, SamProcessor
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def convert_nested_list_to_int(input_list):
    # Use nested list comprehensions to convert all float elements to integers
    return [[[int(num) for num in row] for row in matrix] for matrix in input_list]


def return_pixels(masks,config):

    pixels = []
    iteri = 0
    for i in range(masks.shape[0]):
        
        for x in range(0,masks[i].shape[0]):
            for y in range(0,masks[i].shape[1]):
                if masks[i][x][y] :
                    pixel = {}
                    pixel = {
                        'x': y,
                        'y': x,
                        'color': config["cc"][i],
                        'cluster_label': int(i)
                    }
                    pixels.append(pixel)

    return pixels,config    


def k_means_cluster(image,config):
    """
    Calculate the average of a list of numbers.
    Args:
        image: A numpy function containing the image.
    Returns:
        no_clusters: A dictionary containing results.
    """    
    rows, cols, _ = image.shape
    image_array = np.reshape(image, (rows * cols, -1))

    # Fit K-means clustering
    if "centroids"not in config.keys():
        kmeans = KMeans(n_clusters=config["no_clusters"], random_state=0)
        kmeans.fit(image_array)
        config["centroids"] = kmeans.cluster_centers_
    else:
        kmeans = KMeans(n_clusters=config["no_clusters"], init=config["centroids"], n_init=1)
        kmeans.fit(image_array)
        config["centroids"] = kmeans.cluster_centers_

    # Get cluster labels for all pixels
    labels = kmeans.predict(image_array)

    # Create a list to store pixel information
    pixels = []
    for i, label in enumerate(labels):
        x = i % cols
        y = i // cols
        pixel = {
            'x': x,
            'y': y,
            'color': image[y, x],
            'cluster_label': label
        }
        pixels.append(pixel)

    return pixels, config



###############################################
#           SAM related models                #
###############################################
def k_means_plus_sam(image,config):
    """
    Calculate the average of a list of numbers.
    Args:
        image: A numpy function containing the image.
    Returns:
        no_clusters: A dictionary containing results.
    """
    
    DEVICE = "cpu" #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry["vit_l"](checkpoint="misc/sam_vit_l_0b3195.pth")
    sam.to(device=DEVICE)
    
    mask_predictor = SamPredictor(sam)

    mask_predictor.set_image(image)
    masks, scores, logits = mask_predictor.predict(
        point_coords= np.array([[0,0],[500,0],[500,600],[500,340]]),
        point_labels= np.array([0,1,1,1]),
        multimask_output=True
    )
    num_masks = masks.shape[0]
    import matplotlib.pyplot as plt
    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    plt.axis('off')
    plt.show()    
    print (1/0)
    pixels = {}
    pixels,_ = return_pixels(masks,config)
    return pixels, config

