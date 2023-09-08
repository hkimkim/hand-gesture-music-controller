
"""
Heekyung Kim
CS5330 SP 23
Final Project
This script contains code that preprocesses data and annotation before inputting into network
Reference: https://www.learnpytorch.io/04_pytorch_custom_datasets/#5-option-2-loading-image-data-with-a-custom-dataset
"""

import os
import glob
from pathlib import Path
import torchvision
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import pandas as pd
import xml.etree.ElementTree as ET
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from label_dict import label_index
import matplotlib.patches as patches

# Convert annotation file in XML format to csv
def xml_to_csv(path):
    xml_list = []
    classes_names = []

    for xml_file in list(Path(path).glob('*/*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):

            value = (
                root.find('filename').text,
                member[0].text,
                float(member[5][0].text),
                float(member[5][1].text),
                float(member[5][2].text),
                float(member[5][3].text)
            )
            
            xml_list.append(value)
            classes_names.append(member[0].text)

    column_name = ['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    classes_names = list(set(classes_names))
    classes_names.sort()

    csv_fname = path + "/annotation_file.csv"

    print("Converted XML")
    print(xml_df)
    
    pd.DataFrame.to_csv(xml_df, csv_fname, index=False)
    print(f"Saved to csv file: {csv_fname}")

    return xml_df, classes_names


# Custom data set class
# Has function of ImageFolder() from pytorch
# Images has to be 720 x 1280
# mogrify -resize 1280x720! ./train/background/*.png
class ImageDataset(Dataset):
    def __init__(self, df_annotation, img_dir):
        self.img_annotation = df_annotation
        self.img_dir = img_dir
        self.img_paths = list(Path(img_dir).glob("*/*.png"));
        # Use transformer used for the pretrained model
        self.transform = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT.transforms() 


    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(str(img_path))
        label = self.img_annotation.iloc[idx, 1]

        x_min = self.img_annotation.iloc[idx, 2]
        y_min = self.img_annotation.iloc[idx, 3]
        x_max = self.img_annotation.iloc[idx, 4]
        y_max = self.img_annotation.iloc[idx, 5]
        bbox = [x_min, y_min, x_max, y_max]      

        target = {}
        target["boxes"] = torch.as_tensor([bbox], dtype=torch.float32)
        target["labels"] = torch.as_tensor([label_index[label]["id"]])

        image = self.transform(image)

        return image, target
