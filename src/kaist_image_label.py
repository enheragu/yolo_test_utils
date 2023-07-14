#!/usr/bin/env python3
# encoding: utf-8
"""
    Kaist iterates over Kaist folder plotting label and detection over the images
    taking information about class and oclusion level from XML annotation files.
"""

import untangle
import yaml
import os, errno
from pathlib import Path
import shutil
import cv2 as cv
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


from config_utils import annotation_path, images_path

annotation_path = "/home/quique/umh/kaist_dataset_rgbt/kaist-cvpr15/annotations-xml-new/"
images_path = "/home/quique/umh/kaist_dataset_rgbt/kaist-cvpr15/images/"
labeled_images = "./kaist_labeled_images/"

lwir = "/lwir/"
visible = "/visible/"


class_color = {  'person': (114,196,83), 'person?': (70,133,46), 'cyclist': (26,209,226), 'people': (229,29,46) }

"""
    Process XML entry as:
        <object>
            <name>person</name>
            <bndbox>
                <x>476</x>
                <y>224</y>
                <w>18</w>
                <h>44</h>
            </bndbox>
            <pose>unknown</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <occlusion>0</occlusion>
        </object>
    And returns cv rectangle list to add to imag
"""
def processXML(xml_path, image_path):
    annotation = ()
    image = cv.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not open image {image_path}")

    with open(xml_path) as xml:
        txt_data = ""
        doc = untangle.parse(xml)
        if hasattr(doc.annotation, "object"):
            for object in doc.annotation.object:
                obj_name = object.name.cdata

                start_point = (int(object.bndbox.x.cdata), int(object.bndbox.y.cdata))
                end_point = (int(object.bndbox.x.cdata) + int(object.bndbox.w.cdata), int(object.bndbox.y.cdata) + int(object.bndbox.h.cdata))
                # print(f"Width = {object.bndbox.w.cdata}; Height = {object.bndbox.h.cdata}")
                
                extra_data = "|truncated|" if object.truncated.cdata is True else ""  
                extra_data += "|difficult|" if object.difficult.cdata is True else ""  
                extra_data += "|occlusion|" if object.occlusion.cdata is True else ""
                extra_data = extra_data.replace("||","|")

                # print(f"Found object {obj_name} ({extra_data}) in coord ({start_point}, {end_point})")

                cv.rectangle(image, start_point, end_point, color=class_color[obj_name], thickness=1)

                
                label_str = f"{obj_name} {extra_data}"
                
                # For the text background
                # Finds space required by the text so that we can put a background with that amount of width.
                (w, h), _ = cv.getTextSize(label_str, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # Prints the text.    
                img = cv.rectangle(image, (start_point[0], start_point[1]-h-8), (start_point[0]+w+4, start_point[1]), class_color[obj_name], -1)
                img = cv.putText(image, label_str, (start_point[0]+4, int(start_point[1]-h/2)),
                                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # cv.imshow("img_labeled", image)
    # cv.waitKey(0)
    return image


if __name__ == '__main__':
    
    # Iterate XML folder to process it, gets image associated with the file and 
    # prepare output folder for iages
    for subdir, dirs, files in os.walk(annotation_path):    
        # If its not the lower path with imag just continue
        if len(files) == 0:
            continue

        test_path = subdir.replace(annotation_path, "")
        files_filtered  = list(filter(lambda file: ".xml" in file, files))

        print(f"Process {test_path} dataset:")
        # for file in tqdm(files_filtered):
        def processFile(file):   
            try:
                xml_path = f"{subdir}/{file}"
                # print(xml_path)

                for img_type in (lwir, visible):
                    img_path = images_path + test_path + img_type
                    output_image_path = img_path.replace(images_path,labeled_images)
                    
                    Path(output_image_path).mkdir(parents=True, exist_ok=True)

                    # print(f"Process:\n\t· IMG: {img_path}\n\t· XML: {xml_path}\n\t· Outpu: {output_image_path}")
                    image = processXML(xml_path, img_path  + file.replace(".xml",".jpg"))

                    # Resize imag to save memory
                    scale_percent = 60 # percent of original size
                    width = int(image.shape[1] * scale_percent / 100)
                    height = int(image.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)

                    cv.imwrite(output_image_path + file.replace(".xml","_labeled.jpg"), resized)
                        
            except Exception as e:
                print(f"Exception catched processing {xml_path} with message: {e}")
                raise e
            
        r = process_map(processFile, files_filtered, max_workers = 4, chunksize = 5)

    cv.destroyAllWindows()