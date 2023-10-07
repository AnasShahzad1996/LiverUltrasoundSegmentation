import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Directory containing the images
image_dir = 'path/to/image/directory'

# List all image files in the directory
image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.png')]

# Sort the image files to ensure they are in the correct order
image_files.sort()

# Create a figure for the animation
fig = plt.figure()

# Function to update the frame of the animation
def update(frame):
    plt.clf()  # Clear the previous frame
    img = plt.imread(image_files[frame])  # Read the next image
    plt.imshow(img)  # Display the image
    plt.title(f'Frame {frame+1}/{len(image_files)}')

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(image_files), interval=100)

# Display the animation (this will open a window to view the video)
plt.show()
