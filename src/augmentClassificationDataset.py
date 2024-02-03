
from PIL import Image, ImageEnhance
import os
import random

parent_directory = "DataSets/classification/Data"

def invert_and_save_images(directory):
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
    else:
        for filename in os.listdir(directory):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                with Image.open(os.path.join(directory, filename)) as img:
                    inverted_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    
                    new_filename = filename.split('.')[0] + "_x_axis.png"
                    
                    inverted_img.save(os.path.join(directory, new_filename))
                    print(f"Saved {new_filename}")

def invert_and_save_images_y_axis(directory, axis):
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
    else:
        for filename in os.listdir(directory):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                with Image.open(os.path.join(directory, filename)) as img:
                    if axis == 'y':
                        inverted_img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    else:
                        inverted_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    
                    new_filename = filename.split('.')[0] + f"_{axis}_axis.png"
                    inverted_img.save(os.path.join(directory, new_filename))
                    print(f"Saved {new_filename}")

def add_light_noise_and_save_images(directory, noise_factor=0.2):
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
    else:
        for filename in os.listdir(directory):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                with Image.open(os.path.join(directory, filename)) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    enhancer = ImageEnhance.Brightness(img)
                    img_with_noise = enhancer.enhance(1 + random.uniform(-noise_factor, noise_factor))
                    new_filename = filename.split('.')[0] + "_with_noise.png"
                    img_with_noise.save(os.path.join(directory, new_filename))
                    print(f"Saved {new_filename}")

yes_directory = os.path.join(parent_directory, "YES")
no_directory = os.path.join(parent_directory, "NO")
invert_and_save_images(yes_directory)
invert_and_save_images(no_directory)
invert_and_save_images_y_axis(yes_directory, "y")
invert_and_save_images_y_axis(no_directory, "y")
add_light_noise_and_save_images(yes_directory)
add_light_noise_and_save_images(no_directory)

print("Done")
