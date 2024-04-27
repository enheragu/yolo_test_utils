
from pathlib import Path
import os
import filecmp

home = Path.home()
repo_path = f"{home}/eeha/yolo_test_utils"

yolo_output_path = f"{repo_path}/runs/detect"
filename = "results.yaml"

def compare_files(dir_path):
    # Iterate over the current directory
    search_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            # Find files named 'results.yaml'
            if file == filename:
                search_files.append(os.path.join(root, file))
    
    repeated_dict = {}
    for file in search_files:
        repeated_dict.update(compare_with_similar_files(file, search_files))


    for repeated in repeated_dict.items():
        print(f"Repeated: {repeated[0]} - {repeated[1]}")
        
def compare_with_similar_files(file_path, files):
    # Iterate over the other 'results.yaml' files in the same folder
    repeated_dict = {}
    for other_file in files:
        # Compare the files using filecmp
        if other_file is not file_path: # Dont compare with itself
            if filecmp.cmp(file_path, other_file, shallow=False):
                file1= file_path.replace(yolo_output_path,'').replace(filename,'')
                file2= other_file.replace(yolo_output_path,'').replace(filename,'')
                if file1 < file2:
                    key = file1
                    value = file2
                else:
                    key = file2
                    value = file1
                repeated_dict[key] = value
    
    return repeated_dict

# Call the compare_files function to start the process
compare_files(yolo_output_path)