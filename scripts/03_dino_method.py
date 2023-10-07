# Starting with model DINO
import os
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io, segmentation, color

import torch
import skimage
from PIL import Image
import supervision as sv
import matplotlib.image as mpimg
from matplotlib.widgets import RectangleSelector
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from sklearn.cluster import KMeans
from skimage import io, segmentation, color
from skimage.segmentation import slic
from skimage.color import label2rgb


if __name__ == "__main__":
    import torch.hub

    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    # Load the DINO model onto the device
    vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    vits16.to(device)
    vits16.eval()  # Set the model to evaluation mode

    # Load and process the image
    path = 'datasets/raw_data/patient3/2D/Patient-03-ege-010299.png'
    image_array = cv2.imread(path)
    image_tensor = torch.tensor(image_array.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)

    # Perform forward pass on the GPU
    with torch.no_grad():
        output = vits16(image_tensor)

    # Move the output back to CPU for printing or further processing
    output = output.cpu()

    # Print the output shape
    print("Output shape:", output.shape)