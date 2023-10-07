import os
import cv2
import glob
import pyfiglet
import numpy as np
from PIL import Image
import algorithms as all_algos

 


def get_files(directory):
    """
    Get a list of image file names (PNG, JPEG, etc.) from the specified directory.
    Args:
        directory (str): Path to the directory containing the image files.
    Returns:
        list: List of image file names.
    """
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif']  # Add more extensions as needed
    image_filenames = []

    for ext in image_extensions:
        image_filenames.extend(glob.glob(os.path.join(directory, ext)))
    return image_filenames


def save_image(seg_list,algo,directory,file_name,config,image_np):

    curr_file = file_name.split("/")[-1]
    if seg_list == None:
        return False
    
    if not os.path.exists(directory+"/predict"):
        os.makedirs(directory+"/predict")

    if not os.path.exists(directory+"/predict/"+algo):
        os.makedirs(directory+"/predict/"+algo)

    segmented_image = np.zeros((image_np.shape[0],image_np.shape[1],3),dtype=np.uint8)

    for i in seg_list:
        #print (i)
        segmented_image[i['y'],i['x']] = config['cc'][i['cluster_label']]
    #print (segmented_image)
    image = Image.fromarray(segmented_image)
    image.save(directory+"/predict/"+algo+"/"+curr_file)

    return True

def process_patient(files,algo,config,dir):

    for curr_file in files:
        image_np = np.array(cv2.imread(curr_file))
        curr_func = getattr(all_algos, algo)
        seg_list, config = curr_func(image_np,config)

        save_image(seg_list,algo,dir,curr_file,config,image_np)

if __name__ == "__main__":
    curr_dir = "/home/anas/Desktop/code/practikum/our_code/datasets"
    patients = range(2,12)

    algorithms = ["random_walks"
                  ,"k_means_cluster"
                  ,"voxel_hash"
                  ,"CR_seg"
                  ,"CRF_models"
                  ,"k_means_plus_sam"
                  ,"superpixel_sam"
                  ,"preprogrammed_sam"
                  ,"dino_v1"
                  ,"dino_v2"]
    
    algorithms = ["k_means_plus_sam"]

    config = {
        "no_clusters" : 6
        ,"cc" :{
            0: [0, 0, 0],
            1: [0, 255, 0],
            2: [0, 0, 255],
            3: [255, 165, 0],
            4: [128, 0, 128],
            5: [0, 128, 128]
        }
    }


    for algo in algorithms:
        print ("".join(['#']*100))
        print (pyfiglet.figlet_format(algo,font="slant"))
        print ("".join(['#']*100))


        for patient in patients:
            raw_data_dir = curr_dir + f"/raw_data/patient{patient}/2D"
            raw_data = get_files(raw_data_dir)
            print (f"Patient {patient}. Number of images:",len(raw_data))
            process_patient(raw_data, algo,config,curr_dir + f"/raw_data/patient{patient}")

        # Ending the current algorithm
        print ("".join(['_']*100))

