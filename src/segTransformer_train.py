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

if __name__ == "__main__":
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


    train_dataset.set_transform(train_transforms)
    val_dataset.set_transform(val_transforms)
    test_dataset.set_transform(val_transforms)


    metric = evaluate.load("mean_iou")
    model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)

    training_args = TrainingArguments(
        output_dir="segformer-ultrasound-liver",
        learning_rate=6e-5,
        num_train_epochs=50,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=20,
        eval_steps=20,
        logging_steps=1,
        eval_accumulation_steps=5,
        remove_unused_columns=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()
