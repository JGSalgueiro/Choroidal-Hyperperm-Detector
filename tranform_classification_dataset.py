from PIL import Image
import os

def resize_and_convert_to_grayscale(img_path, output_path, target_size=(480, 480)):
    try:
        img = Image.open(img_path)
        img = img.resize(target_size, Image.ANTIALIAS)
        img = img.convert("L")  # Convert to grayscale
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return False

def process_images(img_folder, mask_folder):
    for root, _, files in os.walk(img_folder):
        for file in files:
            img_path = os.path.join(root, file)
            relative_path = os.path.relpath(img_path, img_folder)
            output_path = os.path.join(mask_folder, relative_path)
            
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Process the image
            success = resize_and_convert_to_grayscale(img_path, output_path)
            if success:
                print(f"Processed: {img_path} -> {output_path}")

if __name__ == "__main__":
    img_folder = "DataSets/classification/img"
    mask_folder = "DataSets/classification/masked"
    process_images(img_folder, mask_folder)
