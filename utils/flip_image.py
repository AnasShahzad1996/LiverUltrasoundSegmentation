import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


folder_path = "/home/anas/Desktop/code/practikum/our_code/datasets/labeled_us_segments/entire_dataset"
folder_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
print (folder_list)

for image_path in folder_list:
    if "label" in image_path:
        pass 
    else:
        image = plt.imread(folder_path+"/"+image_path)
        my_variable = 0



        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.set_title('Press the Up Arrow to Close')
        def on_key(event):
            global my_variable
            if event.key == '1':
                my_variable = 1
                plt.close(fig=fig)
            elif event.key == '2':
                my_variable = 0
                plt.close(fig=fig)
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show()
        print (folder_path+"/"+image_path)
        if my_variable == 0:
            image2 = np.rot90(np.array(image),2)
            image3 = Image.fromarray((image2*255).astype('uint8'))
            image3.save(folder_path+"/"+image_path)

    



