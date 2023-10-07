import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import os
from runpy import run_path
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob
import cv2
import argparse
import numpy as np


def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def add_gaussian_noise(image, std_dev=10):
    # Generate Gaussian noise
    noise = np.random.normal(0, std_dev, image.shape).astype(np.uint8)
    
    # Add noise to the image
    noisy_image = np.clip(image + noise, 0, 255)
    
    return noisy_image

def run_model(inp_dir, out_dir, task):
    os.makedirs(out_dir, exist_ok=True)

    files = natsorted(glob(os.path.join(inp_dir, '*.jpg'))
                    + glob(os.path.join(inp_dir, '*.JPG'))
                    + glob(os.path.join(inp_dir, '*.png'))
                    + glob(os.path.join(inp_dir, '*.PNG')))

    if len(files) == 0:
        raise Exception(f"No files found at {inp_dir}")

    # Load corresponding model architecture and weights
    load_file = run_path(os.path.join(f"/home/anas/Desktop/code/practikum/our_code/datasets/MPRNet/{task}", "MPRNet.py"))
    model = load_file['MPRNet']()
    model.cuda()

    weights = f"/home/anas/Desktop/code/practikum/our_code/datasets/MPRNet/{task}/pretrained_models/model_{task.lower()}.pth"
    load_checkpoint(model, weights)
    model.eval()

    img_multiple_of = 8

    for file_ in files:
        img = Image.open(file_).convert('RGB')
        
        # Add Gaussian noise
        img_array = np.array(img)
        noisy_img_array = add_gaussian_noise(img_array, std_dev=10)
        noisy_img = Image.fromarray(noisy_img_array)

        input_ = TF.to_tensor(noisy_img).unsqueeze(0).cuda()

        # Pad the input if not_multiple_of 8
        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of, ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - h if h % img_multiple_of != 0 else 0
        padw = W - w if w % img_multiple_of != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        with torch.no_grad():
            restored = model(input_)
        restored = restored[0]
        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:, :, :h, :w]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        f = os.path.splitext(os.path.split(file_)[-1])[0]
        save_img((os.path.join(out_dir, f + '.png')), restored)

    print(f"Files saved at {out_dir}")

