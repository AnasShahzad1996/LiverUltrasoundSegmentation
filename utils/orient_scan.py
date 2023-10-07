'''
This function basically sees that in which positin the scan is present
The function basically takes an image and 

i) Rotates
ii) Flips

The image so that it can be used in a downstream task
'''
import os
import cv2
import json
import torch
import skimage
import numpy as np
from PIL import Image
import supervision as sv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io, segmentation, color
from matplotlib.widgets import RectangleSelector
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


# 5 is top and was cyan and can be turned green
# 4 is whitespots is red and now is cyan

COLOR_MACRO = {
    "1" : [255,255,0],
    "2" : [255,0,0],
    "3" : [0,0,255],
    "4" : [0,255,255],
    "5" : [0,255,0],
    "0" : [0,0,0],
}

global_var = ""
label_dict = {}

class ORIENT_SCAN:
    def __init__(self,image_path) -> None:
        self.image_path = image_path
        self.original_image = cv2.imread() 

    def detect_top(self,direction='x-pos'):
        
        return False


def find_line(img):

    indices = []

    if img.shape[2] == 3:
        split1 = np.all(img == [0, 255, 0], axis=-1).astype(np.uint8)
        if len(np.where(split1)[0]) == 0:
            return False
        #plt.imshow(img)
        #plt.show()
        if np.where(split1)[0][0] > int(img.shape[0]/2):
            return False
        else:
            return True
    else:
        print (1/0)

def find_line1(img):

    mask_obj = sam_pred(img,np.array([0,0,img.shape[1],img.shape[0]])).mask[0]
    
    split1 = mask_obj[0:int(img.shape[0]/2),:] 
    split2 = mask_obj[int(img.shape[0]/2):,:] 


    if len(np.where(split1)[0]) < len(np.where(split2)[0]):
        return True
    return False

def which_way_scan1(img,mode):

    if img.shape[1] < img.shape[0]:
        img = np.rot90(img,3)

    if img.shape[1] > img.shape[0]:
        print ("split one")

 

        if mode==0:
            return img
        else:
            return np.rot90(img,2)
    else:
        print (1/0)   

def which_way_scan(img,global_var):

    if img.shape[1] < img.shape[0]:
        img = np.rot90(img,3)

    if img.shape[1] > img.shape[0]:
        print ("split one")

        if find_line(img):
            label_dict[global_var] = 0
            with open("labels.json", "w") as json_file:
                json.dump(label_dict, json_file)

            return img
        else:
            label_dict[global_var] = 1
            with open("labels.json", "w") as json_file:
                json.dump(label_dict, json_file)
            return np.rot90(img,2)
    else:
        print (1/0)    

def real_img(img):
    if img.shape[1] < img.shape[0]:
        img = np.rot90(img,3)

    if img.shape[1] > img.shape[0]:
        print ("split one")

        if find_line1(img):
            label_dict[global_var] = 0
            return img
        else:
            label_dict[global_var] = 1
            with open("labels", "w") as json_file:
                json.dump(label_dict, json_file)
            return np.rot90(img,2)
    else:
        print (1/0)    


def save_seg_map():
    print ("Reading the function")
    folder_path = "/home/anas/Desktop/code/practikum/our_code/datasets/Labeled_us"
    folder_list = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    print (folder_list)

    all_files = []

    for i in range(0,len(folder_list)):
        file_list = os.listdir(folder_path + f"/{folder_list[i]}")
        print ("file list : ",file_list)
        files = [(folder_path + f"/{folder_list[i]}/" + file) for file in file_list if file.lower().endswith('.png')]
        all_files = all_files + files
    #print (all_files)
    sav_dir = "/home/anas/Desktop/code/practikum/our_code/datasets/labeled_us_segments/entire_dataset"

    #all_files = ["/home/anas/Desktop/code/practikum/our_code/datasets/Labeled_us/08-tabea-labels/0_19-labels.png"]
    for i in range(0,len(all_files)):
        curr_path = all_files[i]
        if "-labels.png" in curr_path:
            
            global_var = curr_path
            orig_img = cv2.imread(curr_path,cv2.IMREAD_GRAYSCALE)
            seg_img = np.zeros((orig_img.shape[0],orig_img.shape[1],3))
            print ("curr path : ",curr_path," | ",np.unique(orig_img))
            for x in range(0,orig_img.shape[0]):
                for y in range(0,orig_img.shape[1]):
                    if str(orig_img[x,y]) in COLOR_MACRO.keys():
                        seg_img[x,y] = COLOR_MACRO[str(orig_img[x,y])]
                    else:        
                        seg_img[x,y] = [255,255,255]

            patient_id = curr_path.split("/")[-2].split("-")[0]

            seg_img = which_way_scan(seg_img,global_var)

            save_file = sav_dir +f"/labels_{curr_path.split('/')[-1]}"            
            image = Image.fromarray(seg_img.astype('uint8'))
            image.save(save_file)
        else:
 
            print (curr_path)
            orig_img = cv2.imread(curr_path,cv2.IMREAD_GRAYSCALE)
            patient_id = curr_path.split("/")[-2].split("-")[0]

            import json
            with open("labels.json", "r") as json_file:
                dicti = json.load(json_file)
            mode = 0
            for key in dicti.keys():
                if (curr_path.split("/")[-1]) == key:
                    mode = dicti[key]

            orig_img = which_way_scan1(orig_img,mode)

            save_file = sav_dir +f"/real_{curr_path.split('/')[-1]}"            
            image = Image.fromarray(orig_img.astype('uint8'))
            image.save(save_file)

    import json
    with open("labels", "w") as json_file:
        json.dump(label_dict, json_file)

def sam_pred(img,box):
    '''A SAM model to predict a mask'''
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_l"

    sam = sam_model_registry[MODEL_TYPE](checkpoint="/home/anas/Desktop/code/practikum/our_code/misc/sam_vit_l_0b3195.pth").to(device=DEVICE)
    mask_predictor = SamPredictor(sam)
    image_rgb = img.astype(np.uint8)
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



if __name__=="__main__":
    print ("Orient scan code...")

    save_seg_map()
    print (1/0)
    #path = 'datasets/raw_data/patient3/2D/Patient-03-ege-010299.png'
    #curr_img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    #plt.imshow(which_way_scan(curr_img))
    #plt.show()
    #plt.imshow(curr_img)
    #plt.show()

    #or2 = np.rot90(curr_img,1)
    #plt.imshow(or2)
    #plt.show()
    #plt.imshow(which_way_scan(or2))
    #plt.show()
    #plt.imshow(or2)
    #plt.show()


    #or3 = np.rot90(curr_img,2)     
    #plt.imshow(which_way_scan(or3))
    #plt.show()
    #plt.imshow(or3)
    #plt.show()

    #or4 = np.rot90(curr_img,3)
    #plt.imshow(which_way_scan(or4))
    #plt.show()
    #plt.imshow(or4)
    #plt.show()
    folder_path = "/home/anas/Desktop/code/practikum/our_code/datasets/labeled_us_segments/entire_dataset"
    folder_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    print (folder_list)


    for i in range(0,len(folder_list)):
        curr_path = folder_path + "/" + folder_list[i]
        img = cv2.imread(curr_path)
        correct_img = which_way_scan(img)

        image = Image.fromarray(correct_img.astype('uint8'))
        image.save(curr_path)
