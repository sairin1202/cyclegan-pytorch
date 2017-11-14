import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import skimage.io as io
import glob


def get_images(filename):
    image_names=glob.glob(filename+"*.jpg")
    return image_names

def load_image(file):
    return Image.open(file)

class cyclegan_dataset(Dataset):
    def __init__(self,data_dir1,data_dir2,input_transform=None):
        self.data_dir1=data_dir1
        self.data_dir2=data_dir2
        self.input_transform=input_transform
        self.image1_names=get_images(self.data_dir1)
        self.image2_names=get_images(self.data_dir2)
    def __getitem__(self,index):
        filename1=self.image1_names[index]
        filename2=self.image2_names[index]
        with open(filename1,"rb") as f:
            image1=load_image(f).convert('RGB')
        with open(filename2,"rb") as f:
            image2=load_image(f).convert('RGB')
        

        if self.input_transform is not None:
            image1=self.input_transform(image1)
            image2=self.input_transform(image2)

        return image1,image2

    def __len__(self):
        return len(self.image1_names)
