import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import torchdip

sys.path.insert(1, '/home/anas/Desktop/code/practikum/our_code/datasets/MPRNet')

import my_demo


all_tasks = ['Deblurring', 'Denoising', 'Deraining']


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = image.astype(np.float32) / 255.0       # Normalize to [0, 1]
    return image

# Function to deblur the image using the Deep Image Prior (DIP) model
def deblur_image(image):
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    model = torchdip.DIPDeblur()
    deblurred_tensor = model(image_tensor)
    deblurred_image = deblurred_tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    return deblurred_image

# Function for contrast enhancement using OpenCV
def image_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = np.clip(255 * clahe.apply((255 * image).astype(np.uint8)), 0, 255)
    return enhanced_image.astype(np.float32) / 255.0

def main():
    base_dir = "/home/anas/Desktop/code/practikum/our_code/datasets/raw_data/"
    save_dir = "/home/anas/Desktop/code/practikum/our_code/datasets/pre_processed_images/"
    all_files = ["patient10","patient9"]
    for curr_file in all_files:
        output_dir = f"pre_processed_images/{curr_file}"
        #print ("Denoising images...")
        #my_demo.run_model((base_dir+curr_file),(save_dir+curr_file),all_tasks[0])

        print ("Deblurring images...")
        my_demo.run_model((base_dir+curr_file),(save_dir+curr_file),all_tasks[1])

        print ("Deraining images...")
        my_demo.run_model((base_dir+curr_file),(save_dir+curr_file),all_tasks[2])



if __name__ == "__main__":
    main()





