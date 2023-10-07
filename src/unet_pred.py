import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
import torch.nn.init as init
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io, segmentation, color
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR

# Define the UNet architecture
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()


        self.use_xavier_init = True
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3 ,padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder11 = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3 ,padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder21 = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3 ,padding=1),
            nn.ReLU(inplace=True)
        )
        #self.encoder31 = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)

        '''self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3 ,padding=1),
            nn.ReLU(inplace=True)
        )'''

        self.up31 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up32 = nn.Conv2d(512, 256, kernel_size=3,padding=1)
        self.decoder3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up21 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up22 = nn.Conv2d(256, 128, kernel_size=3,padding=1)
        self.decoder2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up11 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up12 = nn.Conv2d(128, 64, kernel_size=3,padding=1)
        self.last = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )

        self._initialize_weights()

    def forward(self, x):
        # Encoder

        # encoder part
        x1 = self.encoder1(x)
        x11 = self.encoder11(x1)
        x2 = self.encoder2(x11)
        x21 = self.encoder21(x2)
        x3 = self.encoder3(x21)
        #x31 = self.encoder21(x3)
        #x4 = self.encoder4(x31)

        #y3 = self.decoder3(self.up32(torch.cat([self.up31(x4),x3], dim=1)))
        y2 = self.decoder2(self.up22(torch.cat([self.up21(x3),x2], dim=1)))
        y1 = self.last(self.up12(torch.cat([self.up11(y2),x1], dim=1)))

        return y1
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.use_xavier_init:
                    init.xavier_normal_(m.weight)
                else:
                    init.normal_(m.weight, mean=0.0, std=0.02)  # Gaussian initialization
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                if self.use_xavier_init:
                    init.xavier_normal_(m.weight)
                else:
                    init.normal_(m.weight, mean=0.0, std=0.02)  # Gaussian initialization
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class_labels_g = ["Fat deposits", "Top Skin", "Peritoneum", "Liver", "Vessels", "Black background"]

def disp_conf(confusion_matrix_data):
    class_labels = class_labels_g[0:confusion_matrix_data.shape[0]]
    
    confusion_matrix_data = confusion_matrix_data.astype('float') / confusion_matrix_data.sum(axis=1)[:, np.newaxis]
    # Create a figure and axis for the confusion matrix plot
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix_data, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Display class labels (optional)
    if class_labels is not None:
        num_classes = len(class_labels)
        plt.xticks(np.arange(num_classes), class_labels, rotation=45)
        plt.yticks(np.arange(num_classes), class_labels)

    # Display integer values inside the cells
    thresh = confusion_matrix_data.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(confusion_matrix_data[i, j], 'f'),
                    ha="center", va="center",
                    color="white" if confusion_matrix_data[i, j] > thresh else "black")

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()


def metric_function(image1,image2):
    print ("############")
    
    # Calculate Intersection over Union (IoU) and Dice Score
    intersection = np.logical_and(image1 != 0, image2 != 0)
    union = np.logical_or(image1 != 0, image2 != 0)
    iou = np.sum(intersection) / np.sum(union)
    dice_score = (2 * np.sum(intersection)) / (np.sum(image1 != 0) + np.sum(image2 != 0))

    # confusion
    confusion_matrix = np.zeros((6,6))
    for i in range(0,image1.shape[0]):
        for j in range(0,image1.shape[1]):
            x_ax = image1[i,j]
            y_ax = image2[i,j]

            confusion_matrix[x_ax,y_ax] += 1
    # Print the calculated IoU and Dice Score
    print(f'IoU: {iou:.4f}')
    print(f'Dice Score: {dice_score:.4f}')
    return iou,dice_score,confusion_matrix

if __name__ == "__main__":
    print ("UNET prediction")
    parser = argparse.ArgumentParser(description="Transform and process images")
    parser.add_argument("input_dir", type=str, help="Input directory containing your dataset")
    args = parser.parse_args()
    

    model_path = 'unet_model.pth'  # Replace with the path to your pretrained model
    model = UNet(in_channels=1,out_channels=5)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Directory containing images for prediction
    image_dir = f'{args.input_dir}/images'
    pred_path = f'{args.input_dir}/GT'

    list_iou, list_dice = [], []
    total_conf = np.zeros((6,6))


    # Step 3: Perform Predictions
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load and preprocess the image
            image_path = os.path.join(image_dir, filename)
            image_rd = Image.open(image_path)
            image1 = np.array(image_rd)[3:-3,:-6] /255
            image1 = image1.reshape(1,1,680,984)
            print ("image 1 : ",np.unique(image1))
            image2 = torch.from_numpy(image1).float()
            print ("images 1 ",image1.shape)
            

            # Perform prediction
            with torch.no_grad():
                outputs = model(image2)
                #probabilities = torch.softmax(outputs, dim=1)  # Convert to class probabilities
                _, predicted = torch.max(outputs, 1)

            
            image_lab = Image.open(pred_path+"/"+filename)
            image_lab2 = np.array(image_lab)

            curr_iou, curr_dice, curr_conf = metric_function(predicted.detach().numpy()[0],image_lab2[3:-3,:-6])
            list_iou.append(curr_iou)
            list_dice.append(curr_dice)

    total_conf = total_conf + curr_conf
    print ("Average iou : ",(sum(list_iou)/len(list_iou)))
    print ("Average dice : ",(sum(list_dice)/len(list_dice)))
    disp_conf(total_conf)
    disp_conf(total_conf[0:5,0:5])
            
    
