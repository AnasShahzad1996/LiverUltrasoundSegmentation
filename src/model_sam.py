import numpy as np
import cv2
import torch
import numpy as np
from PIL import Image
import supervision as sv
import matplotlib.cm as cm  # Import the colormap module
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import RectangleSelector
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from sklearn.cluster import KMeans
from skimage import io, segmentation, color
from skimage.segmentation import slic
from skimage.color import label2rgb

def bbox2cord(square):
    print (square)

    temp = np.array(square)
    top_point = np.min(temp, axis=0)
    bot_point = np.max(temp, axis=0)

    [width,height] = np.abs(top_point-bot_point) 
    print ("top point : ",top_point)
    print ("bot point : ",bot_point)

    return np.array([top_point[1],top_point[0],top_point[1]+height,top_point[0]+width ])

def predict_linear(point1,point2,predictor):
    grad = (point1[1] - point2[1]) / (point1[0] - point2[0])
    intercept = (grad * -1 * point1[0]) + point1[1]

    pred = (predictor - intercept) / grad 

    return [pred,predictor]


def sam_predict(image,pixels):
    return None



class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = np.array(Image.open(image_path))
        self.image = self.image.T
        self.alpha = 0.2
        self.seg_image = np.zeros(self.image.shape)
        self.bounding_boxes = []
        self.boxes = []
        self.tot_detection = []
        self.segments = None
        self.ex_pix = np.full(self.image.shape, False, dtype=bool)

        self.last = None

    def plot_point(self,point):
        self.axes[0].plot(point[0],point[1],'ro')


    def sam_predictor(self,box):

        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        MODEL_TYPE = "vit_l"

        sam = sam_model_registry[MODEL_TYPE](checkpoint="/home/anas/Desktop/code/practikum/our_code/misc/sam_vit_l_0b3195.pth").to(device=DEVICE)
        mask_generator = SamAutomaticMaskGenerator(sam)

        mask_predictor = SamPredictor(sam)
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        mask_predictor.set_image(image_rgb)

        print ("This is image rgb : ",image_rgb)
        print ("This is the image rgb shape : ",image_rgb.shape)
        print("Data type of the array : ", image_rgb.dtype)

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
        self.tot_detection.append(detections)
        return segmented_image, detections


    def top_skin(self):
        # This function finds the top skin in the ultra-sound
        left_top = [0,10]
        for i in range(0,self.image.shape[1]):
            if self.image[i,10] == 0.0:
                left_top = [i,10]
            else:
                break
        left_bot = [0,20]
        for i in range(0,self.image.shape[1]):
            if self.image[i,20] == 0.0:
                left_bot = [i,20]
            else:
                break

        right_top = [0,10]
        for i in range(self.image.shape[1],-1,-1):
            if self.image[i,10] == 0.0:
                right_top = [i,10]
            else:
                break
        right_bot = [0,20]
        for i in range(self.image.shape[1],-1,-1):
            if self.image[i,20] == 0.0:
                right_bot = [i,20]
            else:
                break

        box1 = bbox2cord([right_bot,predict_linear(right_bot,right_top,int(self.image.shape[1]/5)),left_bot,predict_linear(left_bot,left_top,int(self.image.shape[1]/5))])
        seg_img, det_img = self.sam_predictor(box1)
        return seg_img, det_img

    def part_kmeans(self):
        return None


    ###################################
    # replace with average of pixels
    def avg_pix(self,cluster_img,orig_img):

        new_img = np.zeros(cluster_img.shape)
        avg_color = {}
        for i in range(0,self.k_clusters):
            temp = {"tot":0.0,"tot_pix":0.0}
            avg_color[str(i)] = temp


        for i in range(0,cluster_img.shape[0]):
            for j in range(0,cluster_img.shape[1]):
                avg_color[str(cluster_img[i,j])]["tot"]  += orig_img[i,j] 
                avg_color[str(cluster_img[i,j])]["tot_pix"]  += 1 

        for i in range(0,self.k_clusters):
            avg_color[str(i)]["tot"] = avg_color[str(i)]["tot"]/ avg_color[str(i)]["tot_pix"]

        for i in range(0,cluster_img.shape[0]):
            for j in range(0,cluster_img.shape[1]):
                temp_c = cluster_img[i,j]
                new_img[i,j] = avg_color[str(temp_c)]["tot"]

        print (avg_color)

        return new_img

    def bottom_sack(self):
        self.k_clusters = 5

        original_image = self.image#cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

        height, width  = original_image.shape
        lower_half = original_image
        lower_half_reshaped = lower_half.reshape(-1, 1)

        kmeans = KMeans(n_clusters=self.k_clusters,random_state=0).fit(lower_half_reshaped)
        labels = kmeans.labels_
        clustered_image = labels.reshape(lower_half.shape)

        plt.imshow(clustered_image)
        plt.show()
        mor_img = self.avg_pix(clustered_image,original_image)
        mor_img = mor_img / 255
        mor_img_rgb = np.stack((mor_img, mor_img, mor_img), axis=-1)

        # Define the number of desired superpixels
        num_superpixels = 500  # Adjust this value as needed
        segments = slic(mor_img_rgb, n_segments=num_superpixels, compactness=10)
        segmented_image = label2rgb(segments, mor_img_rgb, kind='avg')

        plt.imshow(segmented_image)
        plt.show()

        temp4 = 80/255
        color_threshold = [temp4,temp4,temp4]

        # Initialize a list to store pixel coordinates
        above_threshold_pixels = []
        # Initialize a binary mask with zeros
        binary_mask = np.zeros_like(mor_img)

        # Iterate through the superpixels and check their color
        for superpixel_label in np.unique(segments):
            # Get the coordinates of all pixels in the current superpixel
            superpixel_coords = np.column_stack(np.where(segments == superpixel_label))

            # Get the average color of the superpixel
            superpixel_color = np.mean(mor_img[segments == superpixel_label], axis=0)

            # Check if the superpixel color is above the threshold
            if (superpixel_color > color_threshold).all() and (superpixel_coords[:, 1] > self.image.shape[1] // 2).all():
                above_threshold_pixels.extend(superpixel_coords)
                binary_mask[segments == superpixel_label] = 1

        return binary_mask

    def vessel_segmentation(self):

        return None

    def process_image(self):
        # Implement image processing logic here
        # You can use popular image processing libraries like OpenCV, Pillow, etc.
        # For example, to load and display the image using Pillow:
        from PIL import Image
        img = Image.open(self.image_path)
        img.show()

    def onselect(self,eclick, erelease):
        if plt.get_current_fig_manager().toolbar.mode == '':
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            self.bounding_boxes.append((x1, y1, x2 - x1, y2 - y1))
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            self.axes[0].add_patch(rect)
            self.fig.canvas.draw()

    def exit_callback(self,event):
        plt.close()

    def exclude_pix(self,mask):
        if mask.shape[0] == 1:
            mask = mask.squeeze(0)
        temp = []
        for i in range(0,mask.shape[0]):
            for j in range(0,mask.shape[1]):
                if mask[i,j] :
                    temp.append([i,j])

        self.ex_pix = np.logical_or(self.ex_pix,mask)
        self.tot_detection.append(temp)
        
    def show_image(self):

        temp2,temp3 = self.top_skin()
        self.exclude_pix(temp3.mask)
        temp3 = self.bottom_sack()
        self.exclude_pix(temp3)
        temp3 = self.vessel_segmentation()


        plt.imshow(self.ex_pix * 1)
        plt.show()


        self.fig, self.axes = plt.subplots(1,2, figsize=(12,6))
        self.axes[0].imshow(self.image,cmap=cm.gray)
        self.axes[0].set_title("Original image")
        self.axes[0].axis("on")

  

        dull_image_array = self.image * self.alpha
        dull_image_array = np.clip(dull_image_array, 0, 255).astype(np.uint8)
        
        self.axes[1].imshow(Image.fromarray(self.ex_pix),cmap=cm.gray)
        self.axes[1].set_title("Segmented image")
        self.axes[1].axis("on")

        rs = RectangleSelector(self.axes[0], self.onselect,useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels')
        exit_button_ax = plt.axes([0.8, 0.01, 0.1, 0.05])
        exit_button = plt.Button(exit_button_ax, 'Exit')
        exit_button.on_clicked(self.exit_callback)
        
        ################
        
        plt.tight_layout()
        plt.show()


if __name__=="__main__":


    image_dir  = "datasets/raw_data/patient3/2D/Patient-03-ege-010303.png"
    curr_obj = ImageProcessor(image_dir)

    curr_obj.show_image()

    print ("Starting with the top skin")

    print ("Vessel prediction")

    print ("Bottom layer for the ultra sound")

