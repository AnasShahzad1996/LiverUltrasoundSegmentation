import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io, segmentation, color

import torch
import time
import skimage
import matplotlib.lines as mlines
from PIL import Image
import supervision as sv
import matplotlib.image as mpimg
from matplotlib.widgets import RectangleSelector
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from sklearn.cluster import KMeans
from skimage import io, segmentation, color
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage import measure
from skimage import filters

k_cl = 4
exc_col = 255


def change_position(image,cursors,y_pos,x_pos, mul):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)

    x_cur = x_pos    
    while x_cur < image.shape[1] and image[y_pos,x_cur] < 5:
        x_cur += mul 
        # Keep the image displayed for an additional 10 seconds
        ax.clear()  # Clear the previous frame
        ax.imshow(image)
        for curr_cur in cursors:
            plt.scatter(curr_cur[1],curr_cur[0],s=500,c='red',marker='x')
        plt.scatter(x_cur,y_pos, s=500, c='red', marker='x')
        plt.pause(0.001) 


    plt.pause(3)
    plt.close()

    return [y_pos,x_cur]

def get_line(point1 ,point2,pred_x):
    

    grad = (point1[1] - point2[1])/(point1[0] - point2[0])
    c_inter = point1[1] - (grad * point1[0])

    new_pred = (grad * pred_x) + c_inter

    last_point = [pred_x,new_pred]
    return last_point

def create_line(image,one_point,two_point):
    # Create the 
    # Create a figure and axis for displaying the image
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)

    # Create a Line2D object to represent the line between the points
    line = mlines.Line2D([one_point[1], two_point[1]], [one_point[0], two_point[0]], color='red')
    ax.add_line(line)
    plt.show()


def get_box(image,excluded_pixels):
    k = 4
    # Create a mask to exclude specified pixels
    height, width = image.shape
    mask = np.ones((height, width), dtype=bool)
    for curr_ex in excluded_pixels:
        mask[curr_ex[0], curr_ex[1]] = False

    # Apply the mask to the grayscale image
    masked_image = image.copy()
    masked_image[~mask] = 0.54  # Set excluded pixels to black (or any other value)

    # Flatten the masked grayscale image into a 1D array
    pixels = masked_image.reshape(-1, 1)

    # Apply k-means clustering to the pixels
    kmeans = KMeans(n_clusters=k_cl, random_state=0).fit(pixels)

    # Get cluster labels for all pixels, including excluded pixels
    cluster_labels = kmeans.labels_

    # Reshape cluster labels to match the image dimensions
    cluster_labels = cluster_labels.reshape((height, width))

    return cluster_labels

def prev():
    print ("Starting with the image...")
    image_path = 'datasets/raw_data/patient3/2D/Patient-03-ege-010299.png'
    original_image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

    height, width = original_image.shape
    print (original_image)

    cursors = []

    cursors.append(change_position(original_image,cursors,5,10,4))
    cursors.append(change_position(original_image,cursors,10,10,4))

    cursors.append(change_position(original_image,cursors,5,width-1,-4))
    cursors.append(change_position(original_image,cursors,10,width-1,-4))

    create_line(original_image, cursors[0],get_line(cursors[0],cursors[1],100))
    create_line(original_image, cursors[2],get_line(cursors[2],cursors[3],100))

def avg_pix(cluster_img,orig_img):

    new_img = np.zeros(cluster_img.shape)
    avg_color = {}
    for i in range(0,k_cl):
        temp = {"tot":0.0,"tot_pix":0.0}
        avg_color[str(i)] = temp


    for i in range(0,cluster_img.shape[0]):
        for j in range(0,cluster_img.shape[1]):
            avg_color[str(cluster_img[i,j])]["tot"]  += orig_img[i,j] 
            avg_color[str(cluster_img[i,j])]["tot_pix"]  += 1 

    for i in range(0,k_cl):
        avg_color[str(i)]["tot"] = avg_color[str(i)]["tot"]/ avg_color[str(i)]["tot_pix"]

    for i in range(0,cluster_img.shape[0]):
        for j in range(0,cluster_img.shape[1]):
            temp_c = cluster_img[i,j]
            new_img[i,j] = avg_color[str(temp_c)]["tot"]

    return new_img

def bot_part(cluster):
    unique_col = np.unique(cluster)
    last_col = unique_col[len(unique_col)-1]

    binary_mask = cluster == last_col

    #plt.imshow(binary_mask)
    #plt.show()

    labeled_image, count = skimage.measure.label(binary_mask, return_num=True)
    objects = skimage.measure.regionprops(labeled_image)
    
    masks_curr = np.zeros(cluster.shape)

    for i in range(0,len(objects)):
        curr_lab = objects[i]["label"]
        #print (objects[i]["area"])
        if objects[i]["area"] > 1000 :
            masks_curr = (labeled_image == curr_lab)
            plt.imshow(masks_curr)
            plt.show()

    print (objects[0]["area"]," | ",objects[0]["label"])

    #plt.imshow(labeled_image)
    #plt.show()

    plt.imshow(masks_curr)
    plt.show()

if __name__=="__main__":
    print ("Create stuff")
    image_path = 'datasets/raw_data/patient3/2D/Patient-03-ege-010299.png'
    original_image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

    height, width = original_image.shape

    clus = get_box(original_image,[])
    avg_clus = avg_pix(clus,original_image)
    print (np.unique(avg_clus))
    bot_part(avg_clus[0:int(height/2),:])