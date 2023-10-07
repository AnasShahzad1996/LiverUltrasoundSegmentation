import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class_labels_g = ["Fat deposits", "Top Skin", "Peritoneum", "Vessel", "Liver", "Black background"]

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

def which_cluster(segment):
    color_list = [[0,255,0],[0,0,255],[255,0,0],[255,255,0],[0,255,255],[0,0,0]]
    for i in color_list:
        if (i[0] == segment[0]) and (i[1] == segment[1]) and (i[2] == segment[2]):
            return color_list.index(i) 
    return 5

def metric_function(path1,path2):
    print ("############")

    # Load the images and segmentation masks (replace with your image and mask paths)
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)
    
    # Calculate Intersection over Union (IoU) and Dice Score
    intersection = np.logical_and(image1 != 0, image2 != 0)
    union = np.logical_or(image1 != 0, image2 != 0)
    iou = np.sum(intersection) / np.sum(union)
    dice_score = (2 * np.sum(intersection)) / (np.sum(image1 != 0) + np.sum(image2 != 0))

    # confusion
    confusion_matrix = np.zeros((6,6))
    for i in range(0,image1.shape[0]):
        for j in range(0,image1.shape[1]):
            x_ax = which_cluster(image1[i,j])
            y_ax = which_cluster(image2[i,j])

            confusion_matrix[x_ax,y_ax] += 1


    # Print the calculated IoU and Dice Score
    print(f'IoU: {iou:.4f}')
    print(f'Dice Score: {dice_score:.4f}')
    return iou,dice_score,confusion_matrix


if __name__=="__main__":

    lab_path = "/home/anas/Desktop/code/practikum/our_code/datasets/labeled_us_segments/entire_dataset/labels"
    pred_path = "/home/anas/Desktop/code/practikum/our_code/datasets/labeled_us_segments/predictions"
    lab_files = [f for f in os.listdir(lab_path) if os.path.isfile(os.path.join(lab_path, f))]
    pred_files = [f for f in os.listdir(pred_path) if os.path.isfile(os.path.join(pred_path, f))]

    base_path = ""

    list_iou, list_dice = [], []
    total_conf = np.zeros((6,6))


    for i in range(0,len(lab_files)):

        pat_name = (lab_files[i][:-11])[7:]
        
        for j in range(0,len(pred_files)):
            pred_name = (pred_files[j][5:])[:-12]
            if pred_name == pat_name:
                print (pat_name," | ",pred_name)
                curr_iou, curr_dice, curr_conf = metric_function(lab_path+"/"+lab_files[i],pred_path+"/"+pred_files[j])
                list_iou.append(curr_iou)
                list_dice.append(curr_dice)
                total_conf = total_conf + curr_conf
                break
    

    print ("Average iou : ",(sum(list_iou)/len(list_iou)))
    print ("Average dice : ",(sum(list_dice)/len(list_dice)))
    disp_conf(total_conf)
    disp_conf(total_conf[0:5,0:5])

#Average iou :  0.5858458790602282
#Average dice :  0.7188521104160165


#Average iou :  0.6512509755886068
#Average dice :  0.7693061706390704
