
from PIL import Image
import os

def crop_image(image_path, output_path):
    with Image.open(image_path) as img:
        left = 75
        upper = 500
        right = img.width - 75
        lower = 1184
        cropped_img = img.crop((left, upper, right, lower))
        resized_img = cropped_img.resize((684, 684))
        resized_img.save(output_path, "PNG")

def tif_to_png(tif_path):
    folder_name = os.path.splitext(os.path.basename(tif_path))[0]
    folder_path = os.path.join("Datasets/OCTBs", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    #----------------------------------------------------------------------------------------------#
    # You need to crop the Tif images accordingly, for some reason height is not normalized        #
    #----------------------------------------------------------------------------------------------#
    # some use 100 -> 2100
    # pinho use 100 -> 3000

    left = 0
    upper = 500
    #right = 834
    right = 834
    lower = 2500

    with Image.open(tif_path) as img:
        for i, page in enumerate(range(img.n_frames)):
            img.seek(page)
            png_path = os.path.join(folder_path, f"{i}.png")
            cropped_img = img.crop((left, upper, right, lower)).resize((500, 500))
            cropped_img.save(png_path, "PNG")
    print("Image conversion complete!")

tif_file = "Datasets/TIFs/OD.tif"
tif_to_png(tif_file)

