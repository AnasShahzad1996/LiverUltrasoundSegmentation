import cv2
import torch
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.image as mpimg





def which_cluster(segment):
    colors={  0: [0, 0, 0],
        1: [0, 255, 0],
        2: [0, 0, 255],
        3: [255, 165, 0],
        4: [128, 0, 128],
        5: [0, 128, 128]}
    
    return colors[0]

def segmented_image(image,segments):
    
    alpha = 0.2
    colored_img = np.zeros(image.shape)
    for curr_segment in segments:
        curr_color = which_cluster(curr_segment)
        for i in range(0,curr_segment.shape[0]):
            for j in range(0,curr_segment.shape[1]):
                colored_img[i,j] = curr_color
    colored_img = (colored_img * alpha) + (colored_img * (1-alpha))
    return colored_img

def left_tri(image):
    #top triangle
    left_top = [0,0]
    for i in range(0,image.shape[0]):
        if image[i,0] == 0.0 :
            left_top = [i,0]
        else:
            break
    bott = [0,0]
    for j in range(0,image.shape[1]):
        if image[0,j] == 0.0:
            bott = [0,j]
        else:
            break

    left_top.reverse()

    return left_top,bott


def tri_create(image):

    left_top = [0,10]
    for i in range(0,image.shape[1]):
        if image[10,i] == 0.0:
            left_top = [i,10]
        else:
            break


    lower_top = [0,15]
    for i in range(0,image.shape[1]):
        if image[15,i] == 0.0:
            lower_top = [i,15]
        else:
            break
    
    grad = (left_top[1] - lower_top[1]) / (left_top[0] - lower_top[0])
    intercept = (grad * -1 * left_top[0]) + left_top[1]

    x_t = 0
    y_pred = (grad * x_t) + intercept
    last_one = [x_t,y_pred]

    tot_pix = []

    for x in range(0,image.shape[1]):
        pred_y = (grad * x) + intercept
        for y in range(0,image.shape[0]):
            if y > pred_y:
                break

            tot_pix.append([x,y])


    

    return left_top,lower_top,last_one, tot_pix#lower_top

def main():

    segments = []
    pixels2seg = []
    image = mpimg.imread("datasets/raw_data/patient3/2D/Patient-03-ege-010303.png")
    print (image.shape)
    
    # get middle pixels to execute


    # get top pixels

    # get bottom pixels
        # Get the white outlines
        # Get the rest of the outlines

    # get pockets of black pixels in between.
    fix, axes = plt.subplots(1,2, figsize=(12,6))

    axes[0].imshow(image)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    #tri_angular =left_tri(image)
    line_color = (1, 0, 0)  # Red color
    #point1_image1 = (375, 000)  # Replace with your desired coordinates for image 1
    #point2_image1 = (000, 600)  # Replace with your desired coordinates for image 1
    #point1_image1, point2_image1 = left_tri(image)
    #axes[0].plot([point1_image1[0], point2_image1[0]], [point1_image1[1], point2_image1[1]], color=line_color, linewidth=2)
    axes[0].plot(10,10,'ro')
    new_point1, new_point2, new_point3, add_pix = tri_create(image)
    axes[0].plot(new_point1[0],new_point1[1],'ro')
    axes[0].plot(new_point2[0],new_point2[1],'ro')
    axes[0].plot(new_point3[0],new_point3[1],'ro')


    point3 = (625,000)
    point4 = (990,550)
    axes[0].plot([point3[0],point4[0]],[point3[1],point4[1]],color=line_color,linewidth=2)

    segment = segmented_image(image,segments)

    axes[1].imshow(segment)
    axes[1].set_title("Segmented image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()