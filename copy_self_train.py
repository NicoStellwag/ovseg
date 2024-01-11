import shutil
import os
from tqdm import tqdm
def copy_directory(source, target):
    # Check if the target directory already exists
    if not os.path.exists(target):
        # If not, create the target directory
        os.makedirs(target)
    
    # Copy the contents of the source directory to the target directory
    shutil.copytree(source, os.path.join(target, os.path.basename(source)))

# Example usage:
source_directory = '/mnt/hdd/self_training_initial/'
target_directory = '/mnt/hdd/self_training_initial2/'


source_names = os.listdir(source_directory)
target_names = os.listdir(target_directory)

for name in tqdm(source_names):
    if not os.path.exists(target_directory + name) and ".pkl" not in name:
        print(source_directory + name,target_directory + name)
        copy_directory(source_directory + name,target_directory)
        #break