import cv2
import numpy as np
import os

mask_dir = "DataSets/ChoroidSegmentation/thickness_masks"
image_dir = "DataSets/ChoroidSegmentation/thickness_map"

output_dir = "DataSets/ChoroidSegmentation/output"
os.makedirs(output_dir, exist_ok=True)

mask_files = os.listdir(mask_dir)

for mask_file in mask_files:
    if True:
        image_name = mask_file #.replace("_seg.png", ".png")

        mask_path = os.path.join(mask_dir, mask_file)
        image_path = os.path.join(image_dir, image_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        mask = cv2.resize(mask, (684, 684))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        border_color = (0, 0, 255)  # Red color
        border_thickness = 2  # Adjust this value to change the thickness of the border
        cv2.drawContours(image_bgr, contours, -1, border_color, border_thickness)

        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, image_bgr)

        print(f"Processed {image_name}.")

print("Image processing completed.")
