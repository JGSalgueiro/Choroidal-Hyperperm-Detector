
from PIL import Image
import os

def calculate_average_color(image_path1, image_path2):
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    if image1.size != image2.size:
        raise ValueError("Images must have the same size.")

    image1 = image1.convert("RGB")
    image2 = image2.convert("RGB")

    average_image = Image.blend(image1, image2, alpha=0.5)

    return average_image

folder_path = "DataSets/ChoroidSegmentation/fuse"
image1_filename = "1.png"
image2_filename = "2.png"

image1_path = os.path.join(folder_path, image1_filename)
image2_path = os.path.join(folder_path, image2_filename)

average_image = calculate_average_color(image1_path, image2_path)

average_image.save(os.path.join(folder_path, "average.png"))

print("Average image saved successfully.")