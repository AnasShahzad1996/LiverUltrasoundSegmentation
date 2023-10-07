import os
import sys
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from SAM_METHOD import SAM_PRED


def sam_method(path,output_dir,file=False):

    if file:
        if "real" in path:
            print ("#################################")
            print (f"Using SAM to predict {path}")
            print ("#################################")

            seg_obj = SAM_PRED(path)

            # Start with getting the k-means cluster
            seg_obj.k_means()
            seg_obj.avg_pix()

            # Then start with getting the shape of the liver if present
            seg_obj.black_space()
            liver_present = seg_obj.liver_p

            # Detecting the top skin if present
            if liver_present:
                seg_obj.top_skin()
                seg_obj.bot_skin()
                seg_obj.exclude_pix()

                seg_obj.vessel_detection()
                seg_obj.exclude_pix()
                seg_obj.liver_segmentation()
                seg_obj.white_segmentation()
                final_seg_img = seg_obj.display_masks()

                save_path = output_dir + "predict.png"

                image = Image.fromarray(final_seg_img.astype('uint8'))
                image.save(save_path)

                print ("#################################")
                print (f"Save image as {save_path}")
                print ("#################################")

    else:
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        #files = ["real_Patient-15-farid-01_33.png"]
        for curr_file in files:
            if "real" in curr_file:
                print ("#################################")
                print (f"Using SAM to predict {curr_file}")
                print ("#################################")
            
                seg_obj = SAM_PRED(path+"/"+curr_file)
        
                # Start with getting the k-means cluster
                seg_obj.k_means()
                seg_obj.avg_pix()

                # Then start with getting the shape of the liver if present
                seg_obj.black_space()
                liver_present = seg_obj.liver_p
                #plt.imshow(seg_obj.original_image)
                #plt.show()
                # Detecting the top skin if present
                if liver_present:
                    seg_obj.top_skin()
                    seg_obj.bot_skin()
                    seg_obj.exclude_pix()

                    seg_obj.vessel_detection()
                    seg_obj.exclude_pix()
                    seg_obj.liver_segmentation()
                    seg_obj.white_segmentation()
                    seg_obj.display_masks()
                    final_seg_img = seg_obj.final_segmented_image

                    save_path = output_dir + curr_file.split(".")[0] + "_predict.png"

                    image = Image.fromarray(final_seg_img.astype('uint8'))
                    image.save(save_path)

                    print ("#################################")
                    print (f"Save image as {save_path}")
                    print ("#################################")

        
        return None

if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <file-folder> <output_directory>")
        sys.exit(1)

    path = sys.argv[1]
    output_directory = sys.argv[2]

    if os.path.isdir(path) or os.path.isfile(path):
        if os.path.isdir(output_directory)==False:
            print (f"Output directory {output_directory} does not exist!")
            pass        
        if os.path.isfile(path):
            sam_method(path,output_directory,file=True)            
        else:
            sam_method(path,output_directory)
    else:
        print (f"Input directory {path} does not exist!")

    
