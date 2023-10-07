import os
import gc
import cv2
import json
import torch
import argparse
import evaluate
import numpy as np
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from datasets import Dataset
from datasets import load_dataset
from torchvision import transforms
from transformers import AutoImageProcessor
from sklearn.metrics import accuracy_score
from transformers import AutoImageProcessor
from torchvision.transforms import ColorJitter
from huggingface_hub import cached_download, hf_hub_url
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dataset(curr_path):
    train_dict, val_dict, test_dict = {"image":[],"annotation":[]},{"image":[],"annotation":[]},{"image":[],"annotation":[]}
    
    #train_dataset
    train_gt    = [f for f in os.listdir(curr_path +"/train/GT") if os.path.isfile(os.path.join(curr_path +"/train/GT", f))]
    train_img   = [f for f in os.listdir(curr_path +"/train/images") if os.path.isfile(os.path.join(curr_path +"/train/images", f))]
    for i in range(0,len(train_gt)):
        img = Image.open((curr_path + "/train/images/"+train_img[i])).convert('RGB')
        gt  = Image.open((curr_path + "/train/GT/"+train_gt[i]))
        train_dict["image"].append(img)
        train_dict["annotation"].append(gt)

    #val_dataset
    val_gt    = [f for f in os.listdir(curr_path +"/validate/GT") if os.path.isfile(os.path.join(curr_path +"/validate/GT", f))]
    val_img   = [f for f in os.listdir(curr_path +"/validate/images") if os.path.isfile(os.path.join(curr_path +"/validate/images", f))]
    for i in range(0,len(val_gt)):
        img = Image.open((curr_path + "/validate/images/"+val_img[i])).convert('RGB')
        gt  = Image.open((curr_path + "/validate/GT/"+val_gt[i]))
        val_dict["image"].append(img)
        val_dict["annotation"].append(gt)

    #test_dataset
    test_gt    = [f for f in os.listdir(curr_path +"/test/GT") if os.path.isfile(os.path.join(curr_path +"/test/GT", f))]
    test_img   = [f for f in os.listdir(curr_path +"/test/images") if os.path.isfile(os.path.join(curr_path +"/test/images", f))]
    for i in range(0,len(test_gt)):
        img = Image.open((curr_path + "/test/images/"+test_img[i])).convert('RGB')
        gt  = Image.open((curr_path + "/test/GT/"+test_gt[i]))
        test_dict["image"].append(img)
        test_dict["annotation"].append(gt)

    train_dataset   = Dataset.from_dict(train_dict)
    val_dataset     = Dataset.from_dict(val_dict)
    test_dataset    = Dataset.from_dict(test_dict)

    return train_dataset,val_dataset,test_dataset

def compute_metrics(eval_pred):
    print ("debugging 1 : ",eval_pred.predictions.shape,"|",eval_pred.label_ids)
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        print ("debugging 2: ",logits.shape,"|",labels.shape)
        # Move logits_tensor and labels to the GPU
        logits_tensor = logits_tensor.to(device)

        # Resize logits_tensor on the GPU and compute argmax in one step
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        print ("debugging 3")

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=num_labels,
            ignore_index=255,
            reduce_labels=False,
        )
        print ("debugging 4")
        for key, value in metrics.items():
            if type(value) is np.ndarray:
                metrics[key] = value.tolist()
        print ("debugging 5")
        return metrics

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

class_labels_g = ["Background","Liver","Vessel","Peritoneum","Fat deposits","Top skin"]

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


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Transform and process images")
    parser.add_argument("input_dir", type=str, help="Input directory containing your dataset")
    args = parser.parse_args()
    
    print("Starting with the transformer model...")
    train_dataset, val_dataset, test_dataset = create_dataset(args.input_dir)

    repo_id = "huggingface/label-files"
    filename = "ade20k-id2label.json"
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    checkpoint = "nvidia/mit-b0"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint, reduce_labels=True)
    jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

    def train_transforms(example_batch):
        images = [jitter(x) for x in example_batch["image"]]
        labels = [x for x in example_batch["annotation"]]
        inputs = image_processor(images, labels)
        return inputs

    def val_transforms(example_batch):
        images = [x for x in example_batch["image"]]
        labels = [x for x in example_batch["annotation"]]
        inputs = image_processor(images, labels)
        return inputs


    #train_dataset.set_transform(train_transforms)
    #val_dataset.set_transform(val_transforms)
    #test_dataset.set_transform(train_transforms)
    print (test_dataset)

    model_directory = 'segformer-ultrasound-liver/checkpoint-14900'
    model = AutoModelForSemanticSegmentation.from_pretrained(model_directory)
    model.eval()

    list_iou, list_dice = [], []
    total_conf = np.zeros((6,6))

    for i in range(0,len(test_dataset["image"])):
        image = test_dataset["image"][i]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available, otherwise use a CPU
        encoding = image_processor(image, return_tensors="pt")
        pixel_values = encoding.pixel_values.to(device)

        model = model.to(device)
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits.cpu()
        
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = (upsampled_logits.argmax(dim=1)[0]).detach().numpy()
        ann_t = test_dataset["annotation"][i]

        
        curr_iou, curr_dice, curr_conf = metric_function(pred_seg,np.array(ann_t))
        list_iou.append(curr_iou)
        list_dice.append(curr_dice)
        total_conf = total_conf + curr_conf

    print ("Average iou : ",(sum(list_iou)/len(list_iou)))
    print ("Average dice : ",(sum(list_dice)/len(list_dice)))
    disp_conf(total_conf)
    disp_conf(total_conf[0:5,0:5])
