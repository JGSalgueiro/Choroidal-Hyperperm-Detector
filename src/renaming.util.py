import os

folder_path = "DataSets/ChoroidSegmentation/thickness_map"

for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        #index = filename.split("_z")[1]  # Extract the index with "z" prefix
        index = filename.split(".")[0]
        if int(index) < 10:
            new_name = "00" + index + "_seg.png"
        elif int(index) < 100:
            new_name = "0" + index + "_seg.png"
        else:
            new_name = index + "_seg.png"
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))

file_list = sorted(os.listdir(folder_path))
num_images = len(file_list)
delete_range = range(75)

for i in delete_range:
    os.remove(os.path.join(folder_path, file_list[i]))

for i in reversed(delete_range):
    os.remove(os.path.join(folder_path, file_list[num_images - 1 - i]))
