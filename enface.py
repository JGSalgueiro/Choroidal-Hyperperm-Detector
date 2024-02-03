import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter, zoom
from skimage import exposure

def calculate_mip(image_path, mask_path):
    image = Image.open(image_path).convert('L')
    mask = Image.open(mask_path).convert('L')

    image = image.resize((684, 684))
    mask = mask.resize((684, 684))

    image_array = np.array(image)
    mask_array = np.array(mask)

    white_pixels = np.where(mask_array == 255, image_array, np.inf)

    mip_intensity = np.min(white_pixels, axis=0)

    return mip_intensity

def calculate_Max_Pixel_Proj(image_path, mask_path):
    image = Image.open(image_path).convert('L')
    mask = Image.open(mask_path).convert('L')

    image = image.resize((684, 684))
    mask = mask.resize((684, 684))

    image_array = np.array(image)
    mask_array = np.array(mask)

    white_pixels = np.where(mask_array == 255, image_array, 0)

    mip_intensity = np.max(white_pixels, axis=0)

    return mip_intensity

def calculate_median_intensity(image_path, mask_path):
    image = Image.open(image_path).convert('L')
    mask = Image.open(mask_path).convert('L')

    image = image.resize((684, 684))
    mask = mask.resize((684, 684))

    image_array = np.array(image)
    mask_array = np.array(mask)

    white_pixels = np.where(mask_array == 255, image_array, 0)

    median_intensity = np.median(white_pixels, axis=0)

    return median_intensity


def calculate_average_intensity(image_path, mask_path):
    image = Image.open(image_path).convert('L')
    mask = Image.open(mask_path).convert('L')

    image = image.resize((684, 684), resample=Image.BICUBIC)
    mask = mask.resize((684, 684), resample=Image.BICUBIC)
    image_array = np.array(image)
    mask_array = np.array(mask)

    white_pixels = np.where(mask_array == 255, image_array, 0)

    average_intensity = np.mean(white_pixels, axis=0)

    return average_intensity

def measure_thickness(mask_path):
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize((684, 684), resample=Image.BICUBIC)
    mask_array = np.array(mask)
    thickness = np.sum(mask_array == 255, axis=0)

    return thickness

def create_intensity_map(image_folder, mask_folder, smoothing_sigma=0.5, median_size=2):
    intensity_matrix = []

    for filename in os.listdir(image_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(image_folder, filename)
            mask_path = os.path.join(mask_folder, filename)
            average_intensity = calculate_average_intensity(image_path, mask_path)  # test with median or avg
            intensity_matrix.append(average_intensity)

    if len(intensity_matrix) == 0:
        raise ValueError("No valid images found in the folder.")

    intensity_map = np.array(intensity_matrix)
    smoothed_intensity_map = gaussian_filter(intensity_map, sigma=smoothing_sigma)
    denoised_intensity_map = median_filter(smoothed_intensity_map, size=median_size)
    gamma = 1.2  # Adjust this value to control the gamma correction

    # Apply gamma correction
    intensity_map_gamma = exposure.adjust_gamma(denoised_intensity_map, gamma=gamma)

    intensity_map_gamma_flipped = np.flip(intensity_map_gamma, axis=0)
    intensity_map_gamma_flipped = np.flip(np.flip(intensity_map_gamma_flipped, axis=1), axis=0)

    return intensity_map_gamma_flipped

def create_thickness_map(mask_folder, smoothing_sigma=1.0, median_size=4):
    intensity_matrix = []

    for filename in os.listdir(mask_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            mask_path = os.path.join(mask_folder, filename)
            average_intensity = measure_thickness(mask_path)  # test with median or avg
            intensity_matrix.append(average_intensity)

    if len(intensity_matrix) == 0:
        raise ValueError("No valid images found in the folder.")

    intensity_map = np.array(intensity_matrix)
    smoothed_intensity_map = gaussian_filter(intensity_map, sigma=smoothing_sigma)
    denoised_intensity_map = median_filter(smoothed_intensity_map, size=median_size)


    flipped_intensity_map = np.flip(denoised_intensity_map, axis=1)

    return flipped_intensity_map

image_folder = 'DataSets/ChoroidSegmentation/thickness_map'
mask_folder = 'DataSets/ChoroidSegmentation/thickness_masks'

try:
    plt.figure()
    intensity_map = create_intensity_map(image_folder, mask_folder)
    plt.imshow(intensity_map, cmap='gray')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("1.png", bbox_inches = 'tight',
        pad_inches = 0)
    plt.show()


    plt.figure()
    thickness_map = create_thickness_map(mask_folder)
    normalized_thickness = (thickness_map - np.min(thickness_map)) / np.ptp(thickness_map)
    color_map = plt.get_cmap('jet')
    thickness_color_map = color_map(normalized_thickness)
    plt.imshow(thickness_color_map)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("2.png", bbox_inches = 'tight',
        pad_inches = 0)
    plt.show()

    print(np.median(thickness_map))
    print(np.mean(thickness_map))
    print(np.amin(thickness_map))
    print(np.amax(thickness_map))

    plt.figure()
    color_map = plt.get_cmap('jet')
    #12x12 
    #15x15 -> 120 vmax
    plt.imshow(thickness_map, cmap=color_map, vmin=0, vmax=120)
    plt.gca().set_axis_on()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("2.png", bbox_inches = 'tight',
        pad_inches = 0)
    plt.show()



except ValueError as e:
    print(e)
