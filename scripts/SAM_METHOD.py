# Starting with model SAM(segment anything model) modified
import os
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io, segmentation, color

import torch
import skimage
import argparse
from PIL import Image
import supervision as sv
import matplotlib.image as mpimg
from matplotlib.widgets import RectangleSelector
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from sklearn.cluster import KMeans
from skimage import io, segmentation, color
from skimage.segmentation import slic
from skimage.color import label2rgb

class SAM_PRED:
    def __init__(self, image_path):
        # Constructor - initialize instance variables
        self.image_path = image_path

        self.original_image = cv2.imread(self.image_path,cv2.IMREAD_GRAYSCALE)
        self.segmented_image = np.zeros(self.original_image.shape)
        self.active_image = np.zeros(self.original_image.shape)

        # hyperparameters : experiment around with them a bit more
        self.no_clusters = 5
        self.exclude_color = 255
        self.black_shadow_background =15
        self.surround_black_thres = 5
        self.global_ex_list = []
        self.mask_list = []

    def get_line(self,point1 ,point2,pred_x):
        grad = (point1[1] - point2[1])/(point1[0] - point2[0])
        c_inter = point1[1] - (grad * point1[0])

        new_pred = (grad * pred_x) + c_inter

        last_point = [pred_x,new_pred]
        return last_point

    def mask2box(self,mask):
        '''Takes a 2-D array of a mask and outputs a box'''        
        indices = np.where(mask)
        min_x = min(indices[0])
        max_x = max(indices[0])

        min_y = min(indices[1])
        max_y = max(indices[1])

        box = np.array([min_y,min_x,max_y,max_x])
        return box

    def mask2Pix(self,mask):
        '''Takes a 2-D array of a mask and outputs a list of pixels'''
        indices = np.where(mask)
        ex_pix = []
        for i in range(0,len(indices[0])):
            ex_pix.append([indices[0][i],indices[1][i]])
        return ex_pix

    def add_exclude_pix(self,ex_list):
        self.global_ex_list = self.global_ex_list + ex_list

    def exclude_pix(self):
        '''Expect a list of pixels to exclude'''
        for curr_pix in self.global_ex_list:
            self.active_image[curr_pix[0],curr_pix[1]] = self.exclude_color

    def avg_pix(self):
        '''
        Averaging the cluster labels
        '''
        new_img = np.zeros(self.segmented_image.shape)
        avg_color = {}
        for i in range(0,self.no_clusters):
            temp = {"tot":0.0,"tot_pix":0.0}
            avg_color[str(i)] = temp


        for i in range(0,self.segmented_image.shape[0]):
            for j in range(0,self.segmented_image.shape[1]):
                avg_color[str(self.segmented_image[i,j])]["tot"]  += self.original_image[i,j] 
                avg_color[str(self.segmented_image[i,j])]["tot_pix"]  += 1 

        for i in range(0,self.no_clusters):
            avg_color[str(i)]["tot"] = avg_color[str(i)]["tot"]/ avg_color[str(i)]["tot_pix"]

        for i in range(0,self.segmented_image.shape[0]):
            for j in range(0,self.segmented_image.shape[1]):
                temp_c = self.segmented_image[i,j]
                new_img[i,j] = avg_color[str(temp_c)]["tot"]
        self.segmented_image = new_img
        self.active_image = new_img
        return None

    def k_means(self):
        '''Performs a k-means clustering of an image into 5 clusters'''
        # Create a mask to exclude specified pixels
        height, width = self.original_image.shape
        mask = np.ones((height, width), dtype=bool)
        for curr_ex in self.global_ex_list:
            mask[curr_ex[0], curr_ex[1]] = False

        # Apply the mask to the grayscale image
        masked_image = self.original_image.copy()
        masked_image[~mask] = 0.54  # Set excluded pixels to black (or any other value)

        # Flatten the masked grayscale image into a 1D array
        pixels = masked_image.reshape(-1, 1)

        # Apply k-means clustering to the pixels
        kmeans = KMeans(n_clusters=self.no_clusters, random_state=0).fit(pixels)

        # Get cluster labels for all pixels, including excluded pixels
        cluster_labels = kmeans.labels_

        # Reshape cluster labels to match the image dimensions
        cluster_labels = cluster_labels.reshape((height, width))

        self.segmented_image = cluster_labels
        self.active_image = cluster_labels
        plt.imshow(self.active_image)
        plt.show()
        return None

    def sam_annotators(self,detections):
        '''Annotating SAM images and boxes'''
        image_rgb = self.original_image.astype(np.uint8)
        box_annotator = sv.BoxAnnotator(color=sv.Color.red())
        mask_annotator = sv.MaskAnnotator(color=sv.Color.red())

        source_image = box_annotator.annotate(scene=image_rgb.copy(), detections=detections, skip_label=True)
        segmented_image = mask_annotator.annotate(scene=image_rgb.copy(), detections=detections)
        return source_image, segmented_image

    def sam_pred(self,box):
        '''A SAM model to predict a mask'''
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        MODEL_TYPE = "vit_l"

        sam = sam_model_registry[MODEL_TYPE](checkpoint="/home/anas/Desktop/code/practikum/our_code/misc/sam_vit_l_0b3195.pth").to(device=DEVICE)
        mask_predictor = SamPredictor(sam)
        image_rgb = self.original_image.astype(np.uint8)
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
        return detections

    def sam_pred_image(self,image,box):
        '''A SAM model to predict a mask'''
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        MODEL_TYPE = "vit_l"

        sam = sam_model_registry[MODEL_TYPE](checkpoint="/home/anas/Desktop/code/practikum/our_code/misc/sam_vit_l_0b3195.pth").to(device=DEVICE)
        mask_predictor = SamPredictor(sam)
        image_rgb = image.astype(np.uint8)
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
        return detections

    def black_space(self):
        ex_pix = []
        self.liver_p = False

        for j in range(0,self.original_image.shape[0]):
            for i in range(0,self.original_image.shape[1]):
                if self.original_image[j,i] < self.black_shadow_background:
                    ex_pix.append([j,i])
                else:
                    break

        for j in range(0,self.original_image.shape[0]):
            for i in range(self.original_image.shape[1]-1,-1,-1):
                if self.original_image[j,i] < self.black_shadow_background:
                    ex_pix.append([j,i])
                else:
                    break

        for j in range(0,self.original_image.shape[1]):
            for i in range(0,self.original_image.shape[0]):
                if self.original_image[i,j] < self.black_shadow_background:
                    ex_pix.append([i,j])
                else:
                    break
                if i > (self.original_image.shape[0]/1.5):
                    self.liver_p = True

        for j in range(0,self.original_image.shape[1]):
            for i in range(self.original_image.shape[0]-1,-1,-1):
                if self.original_image[i,j] < self.black_shadow_background:
                    ex_pix.append([i,j])
                else:
                    break
                if i > (self.original_image.shape[0]/1.5):
                    self.liver_p = True
        
        self.global_ex_list = self.global_ex_list + ex_pix

    def top_skin(self):
        '''
        Find if top skin exists
        Returns:
            1. Mask object (numpy array)
        '''
        cursors = []

        def change_position(image,y_pos,x_pos, mul):
            x_cur = x_pos    
            while x_cur < image.shape[1] and image[y_pos,x_cur] < self.black_shadow_background:
                x_cur += mul 
            return [y_pos,x_cur]

        cursors.append(change_position(self.original_image,5,10,4))
        cursors.append(change_position(self.original_image,10,10,4))

        cursors.append(change_position(self.original_image,5,self.original_image.shape[1]-1,-4))
        cursors.append(change_position(self.original_image,10,self.original_image.shape[1]-1,-4))

        left_bot = self.get_line(cursors[0],cursors[1],150)
        right_bot = self.get_line(cursors[2],cursors[3],150)

        box1 = np.array([left_bot[1],cursors[0][0],right_bot[1],150])
        detect1 = self.sam_pred_image(self.original_image,box1)

        self.mask_list.append(detect1.mask[0])
        #plt.imshow(detect1.mask[0])
        #plt.show()

        return detect1.mask
    
    def bot_skin(self):
        '''
        Find if bottom skin exists
        Returns:
            1. Mask object (numpy array)
        '''
        unique_col = np.unique(self.active_image)
        last_col = unique_col[len(unique_col)-1]

        binary_mask = self.active_image == last_col

        labeled_image, count = skimage.measure.label(binary_mask, return_num=True)
        objects = skimage.measure.regionprops(labeled_image)
        
        masks_curr = np.zeros(self.active_image.shape)

        for i in range(0,len(objects)):
            curr_lab = objects[i]["label"]
            if objects[i]["area"] > 100 :
                masks_curr = np.logical_or(masks_curr,(labeled_image == curr_lab))

        all_ind = np.where(masks_curr)
        bottom_mask = np.zeros(masks_curr.shape)

        # Add additional information about what part i should omit
        ex_pix = []
        for i in range(0,len(all_ind[0])):
            if (all_ind[0][i] > int(self.active_image.shape[0]/2)):
                ex_pix.append([all_ind[0][i],all_ind[1][i]])
                bottom_mask[all_ind[0][i],all_ind[1][i]] = 1
        self.mask_list.append(bottom_mask.astype(bool))
        self.global_ex_list = self.global_ex_list + ex_pix

        return []

    def surround_black(self, segment,real_image):
        '''
        See which vessel is surrounded by black space.
        We exclude those detections that are not within liver segmentation
        '''
        listind = [-1,1]
        indices = np.where(segment)
        iteri = 0
        total = 0

        for curr_ind in range(0,len(indices[0])):
            for i in listind:
                for j in listind:
                    total += 1
                    if real_image[indices[0][curr_ind]+i][indices[1][curr_ind]+j] == self.exclude_color:
                        iteri += 1
                        return False
        if iteri > self.surround_black_thres:
            return False
        
        return True

    def vessel_detection(self):
        '''Vessel detection and refinement using SAM'''
        clustered_image = self.active_image
        unique_col = np.unique(clustered_image)
        last_col = np.sort(unique_col)[0]    

        labeled_image, count = skimage.measure.label((clustered_image == last_col), return_num=True)
        objects = skimage.measure.regionprops(labeled_image)

        masks_curr = np.zeros(clustered_image.shape)
        vessels_curr = np.zeros(clustered_image.shape)
        for i in range(0,len(objects)):
            curr_lab = objects[i]["label"]
            if objects[i]["area"] > 50 and self.surround_black((labeled_image == curr_lab),clustered_image):
                box = self.mask2box(labeled_image == curr_lab)
                sam_mask = self.sam_pred_image(self.active_image,box)
                vessels_curr = np.logical_or(vessels_curr,sam_mask.mask[0])
                masks_curr = np.logical_or(masks_curr,(labeled_image == curr_lab))
            else:
                masks_curr = np.logical_or(masks_curr,(labeled_image == curr_lab))

        all_ind = np.where(masks_curr)
        ex_pix = []
        for i in range(0,len(all_ind[0])):
            ex_pix.append([all_ind[0][i],all_ind[1][i]])
        self.mask_list.append(vessels_curr)
        
        self.global_ex_list = self.global_ex_list + ex_pix

    def liver_segmentation(self):
        '''Liver segmentation using SAM. '''

        ex_pix = []
        unique_col = np.unique(self.active_image)
        liver_indices = []
        for curr_col in unique_col:
            if (curr_col > 10) and (curr_col < 125):
                liver_indices.append(curr_col)

        liver_mask = np.zeros(self.active_image.shape)
        for curr_col in liver_indices:
            liver_mask = np.logical_or(liver_mask,(self.active_image ==curr_col))

        sam_refined = self.sam_pred_image(self.active_image,self.mask2box(liver_mask))
        self.mask_list.append(sam_refined.mask[0])
    
    def white_segmentation(self):
        '''Segments white stuff inside a liver'''
        white_stuff_col = np.unique(self.active_image)[-2]
        white_mask = self.active_image == white_stuff_col
        self.mask_list.append(white_mask)

    def display_all(self,np_arr):
        '''Display all images side by side'''
        sizer = len(np_arr)
        fig, all_axes = plt.subplots(1, sizer, figsize=(10, 5))
        for i in range(0,sizer):    
            all_axes[i].imshow(np_arr[i])  # You can specify the colormap (cmap) as needed
        plt.show()

    def update_progress(self):
        '''Updates the progress made by our method'''

        sizer = len(self.mask_list) + 1 + 1
        fig,all_axes = plt.subplots(1,sizer, figsize=(10,5))
        all_axes[0].imshow(self.original_image)
        all_axes[1].imshow(self.active_image)
        for i in range(2,sizer):
            all_axes[i].imshow(self.mask_list[i-2])
        plt.show()

    def display_masks(self):
        '''Display masks in one image'''
        height,width = self.active_image.shape
        seg_color_image = np.zeros((height,width,3))

        #plt.imshow(self.original_image)
        #plt.show()

        # 1. 3th image should be the liver
        # 2. 0nd image should be the top skin
        # 3. 1rd image should be the bot skin
        # 4. 2th image should be the vessels
        # 5. 4th image should bbe the white stuff
        seg_color_image[self.mask_list[3]] = [255,255,0]
        seg_color_image[self.mask_list[1]] = [0,0,255]

        seg_color_image[self.mask_list[2]] = [255,0,0]
        seg_color_image[self.mask_list[4]] = [0,255,255]
        seg_color_image[self.mask_list[0]] = [0,255,0]

        self.final_segmented_image = seg_color_image
        #plt.imshow(self.final_segmented_image)
        #plt.show()

    def display_single_image(self,image):
        """Display single image"""
        plt.imshow(image)
        plt.show()

class PRED_FOLDER:
    def __init__(self, folder_path):
        # Constructor - initialize instance variables
        file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    def kmeans_sam_pred(self,path):
        seg_obj = SAM_PRED(path)
        
        # Start with getting the k-means cluster
        seg_obj.k_means()
        seg_obj.avg_pix()

        # Then start with getting the shape of the liver if present
        seg_obj.black_space()
        liver_present = seg_obj.liver_p

        # Detecting the top skin if present
        if liver_present:
            seg_obj.top_skin()
            seg_obj.bot_skin()
            seg_obj.exclude_pix()

            seg_obj.vessel_detection()
            seg_obj.exclude_pix()
            seg_obj.liver_segmentation()
            seg_obj.white_segmentation()

        return seg_obj.final_segmented_image

    def kmeans_dino_pred(self,path):
        seg_obj = DINO_PRED(path)

        return seg_obj.final_segmented_image

    def metrics_prediction(self,path):
        """Predicts the IOU and DICE score of the objects"""
        return None

if __name__ == "__main__":
    print ("Segmenting the image: ")
    parser = argparse.ArgumentParser(description="Transform and process images")
    parser.add_argument("input_image", type=str, help="Input directory containing your dataset")
    args = parser.parse_args()
    
    path = args.input_image
    seg_obj = SAM_PRED(path)
    
    # Start with getting the k-means cluster
    seg_obj.k_means()
    seg_obj.avg_pix()

    # Then start with getting the shape of the liver if present
    seg_obj.black_space()
    liver_present = seg_obj.liver_p

    # Detecting the top skin if present
    if liver_present:
        seg_obj.top_skin()
        seg_obj.bot_skin()
        seg_obj.exclude_pix()

        seg_obj.vessel_detection()
        seg_obj.exclude_pix()
        seg_obj.liver_segmentation()
        seg_obj.white_segmentation()
        #seg_obj.update_progress()
        seg_obj.display_masks()
