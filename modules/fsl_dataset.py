# module for creating paths to generate the FSL dataset

import numpy as np
import zipfile
import collections
import re

class FilipinoSignLanguage:
    def __init__(self, zip_path, data_extension='MOV'):
        self.path = zip_path
        self.extension = data_extension

    def load_fsl(self, selected_classes:int or list) -> dict:
        self.selected_classes = selected_classes
        # zip path to folder and file names
        file_names = list_files_from_zip_path(self.path)
        videos = [f for f in file_names if f.endswith(self.extension)]
        folders = [f for f in file_names if not f.endswith(self.extension)]

        # dictionary: keys = class, values = paths of videos in class
        videos_per_class = get_files_per_class(videos, folders)
        
        # subset of files
        files_subset, class_subset = subset_data(videos_per_class, self.selected_classes)

        # reconstruct and return FSL dataset
        return reconstruct_data(files_subset, class_subset)



############ HELPER FUNCTIONS #################
def list_files_from_zip_path(path):
    """ 
    List the files in each class of the dataset given a path of the zip file
    
    Arguments
    ----------
    path: str like
        path of the zip file saved in the same folder
    
    Returns
    ----------
    file_names:
        list of all files read in the file.zip, including folder names
    """
    file_names = []
    with zipfile.ZipFile(path, 'r') as zip:
        for file_info in zip.infolist():
            file_names.append(file_info.filename)
    return file_names

def get_files_per_class(videos, folders):
    # extract the class number from folder names
    videos_per_class = collections.defaultdict(list)
    for folder in folders:
        class_key = int(re.findall(r'\d+', folder)[0])

        for video in videos:
            if folder in video:
                videos_per_class[class_key].append(video)
    return videos_per_class

def subset_data(files:dict, selected_classes, num_videos=20):
    """ Obtain the subset of the complete dataset (files). 
    If selected_classes is an integer, first (selected_classes) classes are obtained. 
    If selected_classes is a list of integers, classes in selected_classes are obtained. """
    if type(selected_classes) == int:
        class_subset = np.arange(selected_classes)
    else:
        class_subset = np.array(selected_classes)
    files_subset = {clss: files[clss][:num_videos] for clss in np.unique(np.sort(class_subset))}
    return files_subset, np.unique(np.sort(class_subset))

def reconstruct_data(files_subset, class_subset):
    """ Transform data into a dictionary with keys 'class' and values 'video paths' """
    dataset = {'data': [], 'target': []}
    for clss in class_subset:
        [[dataset['data'].append(file) for file in files_subset[clss]]]
        [[dataset['target'].append(clss) for file in files_subset[clss]]]
    return dataset