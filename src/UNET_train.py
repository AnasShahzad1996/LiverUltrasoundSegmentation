import os
import gc
import cv2
import numpy as np
import torch
import argparse
import torch.nn.init as init
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset

def mask_to_one_hot(mask, num_classes):
    """
    Converts a grayscale segmentation mask to a one-hot-encoded tensor.
    
    Args:
        mask (PIL.Image or numpy.ndarray): Grayscale mask image or array.
        num_classes (int): Number of classes in the segmentation task.
    
    Returns:
        torch.Tensor: One-hot-encoded tensor.
    """

    # Initialize the one-hot tensor
    one_hot = np.zeros((num_classes, mask.shape[0], mask.shape[1]))

    # Fill in the one-hot tensor
    for class_idx in range(num_classes):
        one_hot[class_idx][mask == class_idx] = 1

    return one_hot

class CustomSemanticSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_folder = os.path.join(root_dir, "images")
        self.mask_folder = os.path.join(root_dir, "GT")
        self.image_files = os.listdir(self.image_folder)
        self.mask_files = os.listdir(self.mask_folder)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        mask_name = os.path.join(self.mask_folder, self.image_files[idx])
        
        image = torch.tensor(cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)[0:-6,3:-3]) / 255
        image = image.view((1,image.shape[0],image.shape[1]))
        mask = torch.tensor(mask_to_one_hot(cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)[3:-3,:-6],5))

        return image, mask
    
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

# Define the IoU (Intersection over Union) function
def calculate_iou(predicted, target):
    intersection = np.logical_and(target, predicted)
    union = np.logical_or(target, predicted)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_dice_coefficient(predicted, target):
    intersection = np.logical_and(predicted, target)
    dice_coefficient = (2.0 * np.sum(intersection)) / (np.sum(predicted) + np.sum(target))
    return dice_coefficient


def train(path1,path2):

    # Define hyperparameters
    batch_size = 1
    learning_rate = 0.0001
    epochs = 10

    # Create dataset loaders for training and validation
    train_dataset = CustomSemanticSegmentationDataset(path1) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CustomSemanticSegmentationDataset(path2) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize the UNet model
    num_classes = 5  # Replace with the actual number of classes
    model = UNet(in_channels=1, out_channels=num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)  # Gamma controls the decay factor

    # Assuming 'model' is already defined
    model.to(device)
    # Training loop
    best_iou = 0.59
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        gc.collect()
        for batch_idx, (data, target) in enumerate(train_loader):
            gc.collect()
            # Move data and target to the GPU
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            _, predicted = torch.max(output, 1)
            _, predicted_tar = torch.max(target, 1)
            #print ("pred shae ",predicted.shape)
            #accuracy = accuracy_score(predicted_tar[0].cpu().numpy(), predicted[0].cpu().numpy())
            #total_accuracy += accuracy

            # Calculate IoU for this batch
            predicted_np = predicted[0].cpu().numpy()
            target_np = predicted_tar[0].cpu().numpy()
            iou = calculate_iou(predicted_np, target_np)
            total_iou += iou
            dice_coefficient = calculate_dice_coefficient(predicted_np, target_np)
            total_dice += dice_coefficient

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}, IoU: {iou}")

        # Calculate average metrics for the entire epoch
        average_loss = total_loss / len(train_loader)
        average_iou = total_iou / len(train_loader)
        total_dice = total_dice / len(train_loader)
        #average_accuracy = total_accuracy / len(train_loader)

        print(f"Epoch {epoch}, Average Loss: {average_loss}, Average IoU: {average_iou}, Average dice score : {total_dice}")

        # Validation loop (update every 10 images)
        if epoch % 1 == 0:
            model.eval()
            val_total_loss = 0.0
            val_total_iou = 0.0
            val_total_dice = 0.0
            num_val_images = 0

            for val_data, val_target in val_loader:
                val_data, val_target = val_data.to(device), val_target.to(device)
                val_output = model(val_data)
                val_loss = criterion(val_output, val_target)

                # Calculate validation accuracy
                _, val_predicted = torch.max(val_output, 1)
                _, val_predicted_tar = torch.max(val_target, 1)                
                #val_accuracy = accuracy_score(val_predicted_tar[0].cpu(), val_predicted[0].cpu())
                #val_total_accuracy += val_accuracy

                # Calculate validation IoU for this batch
                val_predicted_np = val_predicted[0].cpu().numpy()
                val_target_np = val_predicted_tar[0].cpu().numpy()
                val_iou = calculate_iou(val_predicted_np, val_target_np)
                val_total_iou += val_iou


                dice_coefficient = calculate_dice_coefficient(predicted_np, target_np)
                val_total_dice += dice_coefficient

                val_total_loss += val_loss.item()
                num_val_images += len(val_data)
            scheduler.step()
            gc.collect()
            # Calculate average validation metrics
            val_average_loss = val_total_loss / len(val_loader)
            val_average_iou = val_total_iou / len(val_loader)
            val_total_dice = val_total_dice / len(val_loader)
            #val_average_accuracy = val_total_accuracy / len(val_loader)

            print(f"Validation - Epoch {epoch}, Average Loss: {val_average_loss}, Average IoU: {val_average_iou}, Average dice score : {val_total_dice}")
            # Save the trained model
            if val_average_iou > best_iou:
                print ("Saving model weights")
                torch.save(model.state_dict(), "unet_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform and process images")
    parser.add_argument("train_dir", type=str, help="Train dataset directory")
    parser.add_argument("val_dir", type=str, help="Validation dataset directory")
    args = parser.parse_args()
    train(args.train_dir, args.val_dir)
