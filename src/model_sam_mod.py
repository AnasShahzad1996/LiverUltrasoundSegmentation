# Starting with model SAM(segment anything model) modified
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

k_cl = 4
exc_col = 5

def mask2Pix(detec):
    all_det = []
    for i in range(0,detec.shape[1]):
        for j in range(0,detec.shape[2]):
            if detec[0,i,j] :
                all_det.append([i,j])
    return all_det


def return_mask(orig_image,box):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_l"

    sam = sam_model_registry[MODEL_TYPE](checkpoint="/home/anas/Desktop/code/practikum/our_code/misc/sam_vit_l_0b3195.pth").to(device=DEVICE)
    mask_predictor = SamPredictor(sam)
    image_rgb = orig_image.astype(np.uint8)
    image_rgb = np.stack((image_rgb,image_rgb,image_rgb), axis=-1)
    mask_predictor.set_image(image_rgb)


    masks, _, _ = mask_predictor.predict(
        box=box,
        multimask_output=True
    )
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=masks),
        mask=masks
    )
    detections = detections[detections.area == np.max(detections.area)]

    box_annotator = sv.BoxAnnotator(color=sv.Color.red())
    mask_annotator = sv.MaskAnnotator(color=sv.Color.red())

    source_image = box_annotator.annotate(scene=image_rgb.copy(), detections=detections, skip_label=True)
    segmented_image = mask_annotator.annotate(scene=image_rgb.copy(), detections=detections)

    #plt.imshow(source_image)
    #plt.show()
    #plt.imshow(segmented_image)
    #plt.show()

    all_pix = mask2Pix(detections.mask)
    return all_pix

def return_mask1(orig_image,box):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_l"

    sam = sam_model_registry[MODEL_TYPE](checkpoint="/home/anas/Desktop/code/practikum/our_code/misc/sam_vit_l_0b3195.pth").to(device=DEVICE)
    mask_predictor = SamPredictor(sam)
    image_rgb = orig_image.astype(np.uint8)
    image_rgb = np.stack((image_rgb,image_rgb,image_rgb), axis=-1)
    mask_predictor.set_image(image_rgb)


    masks, _, _ = mask_predictor.predict(
        box=box,
        multimask_output=True
    )
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=masks),
        mask=masks
    )
    detections = detections[detections.area == np.max(detections.area)]

    box_annotator = sv.BoxAnnotator(color=sv.Color.red())
    mask_annotator = sv.MaskAnnotator(color=sv.Color.red())

    source_image = box_annotator.annotate(scene=image_rgb.copy(), detections=detections, skip_label=True)
    segmented_image = mask_annotator.annotate(scene=image_rgb.copy(), detections=detections)

    return detections.mask

def display(orig_image,exc_pix):

    dup_image = np.zeros(orig_image.shape)
    for curr_pix in exc_pix:
        dup_image[curr_pix[0],curr_pix[1]] = 0.5
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

    # Display the first image on the left subplot (ax1)
    ax1.imshow(orig_image, cmap='gray')  # You can specify the colormap (cmap) as needed
    ax1.set_title('Original image')

    # Display the second image on the right subplot (ax2)
    ax2.imshow(dup_image, cmap='gray')  # You can specify the colormap (cmap) as needed
    ax2.set_title('Mask exclude')
    
    image1_color = np.stack((orig_image, orig_image, orig_image), axis=-1)

    # Create a mask color array where yellow is applied where the mask is non-zero
    mask_color = np.zeros_like(image1_color)
    mask_color[dup_image > 0] = [255, 255, 0]  # Yellow color ([R, G, B] = [1, 1, 0])

    # Combine the original image and the masked region
    masked_image = np.maximum(image1_color, mask_color)
    ax3.imshow(masked_image)  # You can specify the colormap (cmap) as needed
    ax3.set_title('super imposed')

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    plt.subplots_adjust(wspace=0.1)
    plt.show()

    return None

def k_means_clus(orig_img, excluded_pixels, k=4):

    # Flatten the image into a list of RGB values
    print ("orig image : ",orig_img.shape)
    height, width = orig_img.shape

    all_pix = np.zeros(orig_img.shape , dtype=bool)
    for curr_pix in excluded_pixels:
        all_pix[curr_pix[0],curr_pix[1]] = True

    # Apply k-means clustering to the remaining pixels
    mid_mask = orig_img * all_pix
    print (mid_mask)
    kmeans = KMeans(n_clusters=k_cl, random_state=0).fit(mid_mask)

    # Get cluster labels for all pixels, including excluded pixels
    cluster_labels = np.full(orig_img.shape, -1)  # Initialize labels with -1
#    cluster_labels[all_pix] = kmeans.labels_  # Assign cluster labels to remaining pixels

    plt.imshow(kmeans.labels_)
    plt.show()

black_shadow = 15

def left_right_ride(orig_img):
    ex_pix = []

    for j in range(0,orig_img.shape[0]):
        for i in range(0,orig_img.shape[1]):
            if orig_img[j,i] < black_shadow:
                ex_pix.append([j,i])
            else:
                break

    for j in range(0,orig_img.shape[0]):
        for i in range(orig_img.shape[1]-1,-1,-1):
            if orig_img[j,i] < black_shadow:
                ex_pix.append([j,i])
            else:
                break

    return ex_pix

def top_down(orig_img):            
    ex_pix = []

    for j in range(0,orig_img.shape[1]):
        for i in range(0,orig_img.shape[0]):
            if orig_img[i,j] < black_shadow:
                ex_pix.append([i,j])
            else:
                break

    for j in range(0,orig_img.shape[1]):
        for i in range(orig_img.shape[0]-1,-1,-1):
            if orig_img[i,j] < black_shadow:
                ex_pix.append([i,j])
            else:
                break

    return ex_pix

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

def k_means(image,excluded_pixels):
    k = k_cl
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

def k_means_col(image,excluded_pixels):
    # Create a mask to exclude specified pixels
    image = np.stack((image, image, image), axis=-1)
    height, width, _ = image.shape
    mask = np.ones((height, width), dtype=bool)
    for curr_pix in excluded_pixels:
        mask[curr_pix[0], curr_pix[1]] = False

    # Apply the mask to the image
    masked_image = image.copy()
    masked_image[~mask] = [255,255,0]  # Set excluded pixels to black (or any other value)

    # Flatten the masked image into a list of RGB values
    pixels = np.reshape(masked_image, (height * width, 3))
    kmeans = KMeans(n_clusters=k_cl, random_state=0).fit(pixels)
    cluster_labels = kmeans.labels_
    cluster_labels = cluster_labels.reshape((height, width))
    return cluster_labels

def display_all(np_arr):

    sizer = len(np_arr)
    fig, all_axes = plt.subplots(1, sizer, figsize=(10, 5))
    for i in range(0,sizer):    
        all_axes[i].imshow(np_arr[i])  # You can specify the colormap (cmap) as needed
    plt.show()

def ex_pix(image,ex_pix):
    #print ("exc pix : ",ex_pix)
    #print ("imager : ",image)
    for p in ex_pix:
        image[p[0],p[1]] = exc_col
    return image

def change_position(image,cursors,y_pos,x_pos, mul):
    x_cur = x_pos    
    while x_cur < image.shape[1] and image[y_pos,x_cur] < exc_col:
        x_cur += mul 
    return [y_pos,x_cur]

def get_line(point1 ,point2,pred_x):
    

    grad = (point1[1] - point2[1])/(point1[0] - point2[0])
    c_inter = point1[1] - (grad * point1[0])

    new_pred = (grad * pred_x) + c_inter

    last_point = [pred_x,new_pred]
    return last_point

def top_skin(image):
    cursors = []

    cursors.append(change_position(original_image,cursors,5,10,4))
    cursors.append(change_position(original_image,cursors,10,10,4))

    cursors.append(change_position(original_image,cursors,5,width-1,-4))
    cursors.append(change_position(original_image,cursors,10,width-1,-4))

    left_bot = get_line(cursors[0],cursors[1],150)
    right_bot = get_line(cursors[2],cursors[3],150)

    box1 = np.array([left_bot[1],cursors[0][0],right_bot[1],150])
    detect1 = return_mask(image,box1)

    return detect1
    #box1 = bbox2cord([right_bot,predict_linear(right_bot,right_top,int(self.image.shape[1]/5)),left_bot,predict_linear(left_bot,left_top,int(self.image.shape[1]/5))])
    #seg_img, det_img = self.sam_predictor(box1)
    #return seg_img, det_img


def bottom_skin(cluster):
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
        if objects[i]["area"] > 1000 :
            masks_curr = np.logical_or(masks_curr,(labeled_image == curr_lab))

    all_ind = np.where(masks_curr)
    ex_pix = []
    for i in range(0,len(all_ind[0])):
        #ex_pix.append([all_ind[1][i],all_ind[0][i]])
        if all_ind[0][i] > int(cluster.shape[0]/2):
            ex_pix.append([all_ind[0][i],all_ind[1][i]])

    #print ("ex_pix : ",masks_curr.shape," | ",masks_curr)
    #print (objects[0]["area"]," | ",objects[0]["label"])
    #print ("ex pix : ",ex_pix)

    #plt.imshow(labeled_image)
    #plt.show()

    #plt.imshow(masks_curr)
    #plt.show()
    return ex_pix

def is_black(segment,real_image):

    indices = np.where(segment)
    
    list2check = []
    for curr_ind in range(0,len(indices[0])):
        for i in range(-1,2):
            for j in range(-1,2):
                if i==0 and j==0:
                    pass
                else:
                    list2check.append([indices[0][curr_ind]+i,indices[1][curr_ind]+j])
    
    # 10% of the vessels
    iteri = 0
    for curr_cord in list2check:
        if real_image[curr_cord[0],curr_cord[1]] == 5:
            iteri += 1

    if (iteri/len(list2check)) >=  0.5:
        return False

    return True

def vessel_detection(clustered_image):
    unique_col = np.unique(clustered_image)
    last_col = np.sort(unique_col)[0]    

    labeled_image, count = skimage.measure.label((clustered_image == last_col), return_num=True)
    objects = skimage.measure.regionprops(labeled_image)

    masks_curr = np.zeros(clustered_image.shape)

    for i in range(0,len(objects)):
        curr_lab = objects[i]["label"]
        if objects[i]["area"] > 50 and is_black((labeled_image == curr_lab),clustered_image):
            masks_curr = np.logical_or(masks_curr,(labeled_image == curr_lab))
            #plt.imshow((labeled_image == curr_lab))
            #plt.show()

    all_ind = np.where(masks_curr)
    ex_pix = []
    for i in range(0,len(all_ind[0])):
        #ex_pix.append([all_ind[1][i],all_ind[0][i]])
        if all_ind[0][i] > int(clustered_image.shape[0]/2):
            ex_pix.append([all_ind[0][i],all_ind[1][i]])

    #print ("ex_pix : ",masks_curr.shape," | ",masks_curr)
    #print (objects[0]["area"]," | ",objects[0]["label"])
    #print ("ex pix : ",ex_pix)

    #plt.imshow(labeled_image)
    #plt.show()

    #plt.imshow(masks_curr)
    #plt.show()
    return ex_pix,masks_curr

def true_mask2pix(mask):
    ex_pix = []
    indices = np.where(mask)

    for i in range(0,len(indices[0])):
        ex_pix.append([indices[0][i],indices[1][i]])

    return ex_pix

def white_segmentation(clustered_image):
    unique_col = np.unique(clustered_image)
    last_col = np.sort(unique_col)[(len(unique_col)-1)]    
    white_mask = clustered_image == last_col

    plt.imshow(white_mask)
    plt.show()

    return white_mask,true_mask2pix(white_mask)

def liver_segmentation(clustered_image):
    unique_col = np.unique(clustered_image)
    print (unique_col)
    last_col = np.sort(unique_col)[2]    
    white_mask = clustered_image == last_col
    plt.imshow(white_mask)
    plt.show()

    plt.imshow((clustered_image == np.sort(unique_col)[3]))
    plt.show()

    return white_mask,true_mask2pix(white_mask)

def mask2box(mask):
    '''Takes a 2-D array of a mask and outputs a box'''        
    indices = np.where(mask)
    min_x = min(indices[0])
    max_x = max(indices[0])

    min_y = min(indices[1])
    max_y = max(indices[1])

    box = np.array([min_x,min_y,max_x,max_y])
    return box


if __name__ == "__main__":
    print ("Starting with sam model...")
    mask_curr = []
    image_path = 'datasets/raw_data/patient3/2D/Patient-03-ege-010299.png'
    original_image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

    height, width = original_image.shape

    print ("Testing image with shape: ",original_image.shape)

    exc_pix = []
    print ("Getting left side and right")
    left_box = np.array([0,0,int(width/2),height])
    right_box = np.array([int(width/2),0,width,height])
    left_pix = return_mask(original_image,left_box)
    right_pix = return_mask(original_image,right_box)

    exc_pix = left_pix
    exc_pix = exc_pix + right_pix


    print ("Removing other clusters")
    #k_means_clus(original_image,exc_pix)


    print ("Printing both of the images side by side : ")
    #display(original_image, exc_pix)

    new_ex = left_right_ride(original_image)
    new_ex = top_down(original_image) + new_ex
    #display(original_image,new_ex)
    #cluster_col = k_means_col(original_image,new_ex)
    cluster_l = k_means(original_image,new_ex)

    mid = avg_pix(cluster_l,original_image)
    mid = ex_pix(mid, new_ex)

    top_l = top_skin(mid)
    mask_curr.append(top_l)
    mid = ex_pix(mid,top_l)

    bot_l = bottom_skin(mid)
    mask_curr.append(bot_l)
    mid = ex_pix(mid,bot_l)

    vessel_mask,other_stuff = vessel_detection(mid)
    mid = ex_pix(mid,vessel_mask)
    mask_curr.append(other_stuff)

    #  white stuff
    white_mask,_ = white_segmentation(mid)
    mid = ex_pix(mid,white_mask)
    mask_curr.append(white_mask)

    #  liver stuff
    liver_mask,_ = liver_segmentation(mid)
    mid = ex_pix(mid,liver_mask)
    mask_curr.append(liver_mask)
    liver_box = mask2box(liver_mask)

    #liver_mask = return_mask1(mid,liver_box)

    #print ("mask curr : ",mask_curr)
    #print ("length of mask curr : ",len(mask_curr))

    display_all([original_image,cluster_l,other_stuff,white_mask,liver_mask])
