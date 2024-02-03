

from PIL import Image
import os
import cv2
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

imgFolder = os.path.abspath(os.path.join(script_dir, "./../DataSets/ChoroidSegmentation/img"))
maskedFolder =  os.path.abspath(os.path.join(script_dir, "./../DataSets/ChoroidSegmentation/masked")) 

noise_type = 'gaussian'
noise_mean = 0
noise_std = 30


for filename in os.listdir(imgFolder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image = Image.open(os.path.join(imgFolder, filename))
        y_flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        new_filename = filename.split('.')[0] + '_yFlipped.' + filename.split('.')[1]
        y_flipped_image.save(os.path.join(imgFolder, new_filename))

for filename in os.listdir(maskedFolder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image = Image.open(os.path.join(maskedFolder, filename))
        y_flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        new_filename = filename.split('.')[0] + '_yFlipped.' + filename.split('.')[1]
        y_flipped_image.save(os.path.join(maskedFolder, new_filename))

for img_filename in os.listdir(imgFolder):
    if img_filename.endswith('.jpg') or img_filename.endswith('.png') or img_filename.endswith('.jpeg'):
        # Read the image
        img_path = os.path.join(imgFolder, img_filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Add the noise
        if noise_type == 'gaussian':
            img_noisy = img + np.random.normal(noise_mean, noise_std, img.shape)

        # Save the noisy image
        img_noisy_filename = os.path.splitext(img_filename)[0] + '_noisy' + os.path.splitext(img_filename)[1]
        img_noisy_path = os.path.join(imgFolder, img_noisy_filename)
        cv2.imwrite(img_noisy_path, img_noisy)

for filename in os.listdir(maskedFolder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image = Image.open(os.path.join(maskedFolder, filename))
        new_filename = filename.split('.')[0] + '_noisy.' + filename.split('.')[1]
        image.save(os.path.join(maskedFolder, new_filename))