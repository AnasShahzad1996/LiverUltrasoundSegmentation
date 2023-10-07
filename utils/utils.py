import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io, segmentation, color

import torch
from PIL import Image
import supervision as sv
import matplotlib.image as mpimg
from matplotlib.widgets import RectangleSelector
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from sklearn.cluster import KMeans
from skimage import io, segmentation, color
from skimage.segmentation import slic
from skimage.color import label2rgb


# Load the image (replace 'image_path.jpg' with your image file)
image_path = 'datasets/raw_data/patient3/2D/Patient-03-ege-010303.png'
image_path = 'datasets/raw_data/patient3/2D/Patient-03-ege-010052.png'
image_path = 'datasets/raw_data/patient3/2D/Patient-03-ege-010299.png'
#image_path = 'datasets/raw_data/patient3/2D/Patient-03-ege-010477.png'

k_clusters = 5

original_image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

height, width  = original_image.shape
lower_half = original_image#[height // 2:, :]
lower_half_reshaped = lower_half.reshape(-1, 1)
print (lower_half)
print (lower_half_reshaped.shape)



kmeans = KMeans(n_clusters=k_clusters,random_state=0).fit(lower_half_reshaped)
labels = kmeans.labels_
clustered_image = labels.reshape(lower_half.shape)


###################################
# replace with average of pixels
def avg_pix(cluster_img,orig_img):

    new_img = np.zeros(cluster_img.shape)
    avg_color = {}
    for i in range(0,k_clusters):
        temp = {"tot":0.0,"tot_pix":0.0}
        avg_color[str(i)] = temp


    for i in range(0,cluster_img.shape[0]):
        for j in range(0,cluster_img.shape[1]):
            avg_color[str(cluster_img[i,j])]["tot"]  += orig_img[i,j] 
            avg_color[str(cluster_img[i,j])]["tot_pix"]  += 1 

    for i in range(0,k_clusters):
        avg_color[str(i)]["tot"] = avg_color[str(i)]["tot"]/ avg_color[str(i)]["tot_pix"]

    for i in range(0,cluster_img.shape[0]):
        for j in range(0,cluster_img.shape[1]):
            temp_c = cluster_img[i,j]
            new_img[i,j] = avg_color[str(temp_c)]["tot"]

    print (avg_color)

    return new_img


def sam_pred(box,orig_image):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_l"

    sam = sam_model_registry[MODEL_TYPE](checkpoint="/home/anas/Desktop/code/practikum/our_code/misc/sam_vit_l_0b3195.pth").to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)

    mask_predictor = SamPredictor(sam)
    #image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    #image_rgb = (orig_image / 255.0).astype(np.float32)
    image_rgb = orig_image.astype(np.uint8)
    image_rgb = np.stack((image_rgb,image_rgb,image_rgb), axis=-1)

    print ("image rgb : ",image_rgb.shape) 
    mask_predictor.set_image(image_rgb)


    masks, scores, logits = mask_predictor.predict(
        box=box,
        multimask_output=True
    )
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=masks),
        mask=masks
    )
    detections = detections[detections.area == np.max(detections.area)]

    box_annotator = sv.BoxAnnotator(color=sv.Color.red())
    mask_annotator = sv.MaskAnnotator(color=sv.Color.red())

    source_image = box_annotator.annotate(scene=image_rgb.copy(), detections=detections, skip_label=True)
    segmented_image = mask_annotator.annotate(scene=image_rgb.copy(), detections=detections)

    return source_image,segmented_image


##################################
print (clustered_image.shape)


mor_img = avg_pix(clustered_image,original_image)

#plt.imshow(mor_img)
#plt.show()

box = None
box = np.array([495,0,990,686])

source_ime ,new_seg_image = sam_pred(box,mor_img)
print (source_ime.shape)
plt.imshow(source_ime)
plt.show()
plt.imshow(new_seg_image)
plt.show()

binary_mask = (mor_img >= 10).astype(np.uint8)

# Display the binary mask as an image
plt.imshow(binary_mask, cmap='gray')
plt.axis('off')
plt.show()