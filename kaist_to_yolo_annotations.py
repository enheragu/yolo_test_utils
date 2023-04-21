#!/usr/bin/env python3
# encoding: utf-8

"""
    Kaist to Yolo annotation formatting: one *.txt file per image (if no objects in image, no *.txt file is required). 
    The *.txt file specifications are:
    · One row per object
    · Each row is class x_center y_center width height format.
    · Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide x_center and 
      width by image width, and y_center and height by image height.
    · Class numbers are zero-indexed (start from 0).
"""
import untangle
import yaml
import os
import shutil

annotation_path = "/home/quique/umh/kaist_dataset_rgbt/kaist-cvpr15/annotations-xml-new/"
images_path = "/home/quique/umh/kaist_dataset_rgbt/kaist-cvpr15/img/"
lwir = "/lwir/"
visible = "/visible/"
label_folder = "/labels/"
images_folder = "/images/"

object_class = {}
obj_index = 0
processed_files = 0

# TO check against default yolo, classes have to match the coco128.yaml
class_data_coco = {  'person': 0,  'cyclist': 80, 'people': 81 } # people does not exist in coco dataset, use 80 as tag


## Move all imags to 'iamge' folder. Just done once to reorder the dataset
# for subdir, dirs, files in os.walk(images_path):
#     # print(subdir)
    
#     for file in files:
#         if ".jpg" in file and images_folder not in subdir:
#             if not os.path.exists(subdir + images_folder): os.makedirs(subdir + images_folder)
#             if not os.path.exists(subdir + label_folder): os.makedirs(subdir + label_folder)
#             file_path = subdir + "/" + file
#             new_file_path = subdir + images_folder + file.replace(".jpg", ".png")
#             shutil.move(file_path, new_file_path)
#             print(f"[{processed_files}]Processed {file_path}")
#             processed_files += 1


processed_files = 0
for subdir, dirs, files in os.walk(annotation_path):
    # print(f"{subdir = }; {dirs = }; {len(files) = }")
    # If its not the lower path with imag just continue
    if len(files) == 0:
        continue

    lwir_label_path = images_path + subdir.replace(annotation_path, "") + lwir + label_folder
    visible_label_path = images_path + subdir.replace(annotation_path, "") + visible + label_folder
    
    # print(f"Procesing images from {subdir}, sotoring annotations to {lwir_path} and {visible_path}.")
    try: 
        print("-------")
        for file in files:
            annotation_file = subdir + "/" + file
            print(f"[{processed_files}] From {annotation_file}")
            processed_files += 1
            out_path = (lwir_label_path + file.replace(".xml", ".txt"), visible_label_path + file.replace(".xml", ".txt"))
            with open(annotation_file) as xml:
                txt_data = ""
                doc = untangle.parse(xml)
                if hasattr(doc.annotation, "object"):
                    for object in doc.annotation.object:
                        obj_name = object.name.cdata.replace("?","")
                        # print(f"Detected {obj_name} in (x,y) = ({object.bndbox.x.cdata},{object.bndbox.y.cdata}) with (w,h) = ({object.bndbox.w.cdata},{object.bndbox.h.cdata})")
                        if obj_name not in object_class:
                            object_class[obj_name] = obj_index
                            obj_index += 1
                        
                        img_width = float(doc.annotation.size.width.cdata)
                        img_height = float(doc.annotation.size.height.cdata)

                        x_centered = float(object.bndbox.x.cdata) + float(object.bndbox.w.cdata) / 2.0
                        y_centered = float(object.bndbox.y.cdata) + float(object.bndbox.h.cdata) / 2.0

                        x_normalized = x_centered / img_width
                        y_normalized = y_centered / img_height
                        w_normalized = float(object.bndbox.w.cdata) / img_width
                        h_normalized = float(object.bndbox.h.cdata) / img_height

                        txt_data += f"{class_data_coco[obj_name]} {x_normalized} {y_normalized} {w_normalized} {h_normalized}\n"
                    # print(out_path)
                    # print(txt_data)
                    for file in out_path:
                        with open(file, 'w+') as output:
                            output.write(txt_data)

        print("-------")
    except Exception as e:
        print(f"Exception catched processing {annotation_file} with message: {e}")

yaml_data_path = "./dataset_config/yolo_obj_classes.yaml"
with open(yaml_data_path, "w+") as file:
    # Swap key and value to access by number later
    yaml_data = {"path": images_path, "train": "#TBD", "val": "#TBD", "test": "#TBD",
                 "names": {v: k for k, v in class_data_coco.items()}}
    yaml.dump(yaml_data, file)

print(f"Dumped data about classes in: {yaml_data_path}. \nData is: \t\n{yaml_data}")
print(f"Processed files: {processed_files}")
