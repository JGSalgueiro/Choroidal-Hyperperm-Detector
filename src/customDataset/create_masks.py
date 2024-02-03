import json
from PIL import Image, ImageDraw

annotationFolder = "DataSets/ChoroidSegmentation/annotation"
imgFolder = "DataSets/ChoroidSegmentation/img"
maskedFolder = "DataSets/ChoroidSegmentation/masked"


with open(annotationFolder + "/via_export_json (2).json") as f:
    annotations = json.load(f)

for image_name, image_data in annotations.items():
    image_path = image_data['filename']
    image = Image.open(imgFolder + "/" + image_path)

    mask = Image.new('1', image.size)

    for region in image_data['regions']:
        points_x = region['shape_attributes']['all_points_x']
        points_y = region['shape_attributes']['all_points_y']
        points = list(zip(points_x, points_y))

        draw = ImageDraw.Draw(mask)
        draw.polygon(points, fill=1)

    mask_path = maskedFolder + "/" + image_name.split("png")[0] + "png"
    mask.save(mask_path)