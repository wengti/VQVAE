
from pathlib import Path
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
import random
import torch

def find_classes(directory):
    
    directory = Path(directory)
    class_names = sorted(entry.name for entry in os.scandir(directory))
    if not class_names:
        raise FileNotFoundError(f"No valid class names can be found in {directory}. Please check.")
    
    class_to_idx = {}
    for idx, name in enumerate(class_names):
        class_to_idx[name] = idx
    
    return class_names, class_to_idx


class custom_dataset(Dataset):
    
    def __init__(self, directory, config):
        directory = Path(directory)
        self.path_list = list(directory.glob('*/*.png'))
        self.classes, self.class_to_idx = find_classes(directory)
        self.config = config
    
    def load_image(self, index):
        img = Image.open(self.path_list[index])
        return img
    
    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, index):
        
        #1. include conditions when there is a 3 channels input
        
        img = self.load_image(index)
        
        simple_transform = transforms.ToTensor()
        img_tensor = simple_transform(img) # 1x28x28, float tensor, range: 0 to 1
        
        if self.config['input_channels'] == 3:
            img_tensor_r = ((img_tensor*255)* np.clip(random.random(), 0.2 , 1)).int() # 1x28x28, int tensor, range: 0-255
            img_tensor_g = ((img_tensor*255)* np.clip(random.random(), 0.2 , 1)).int() # 1x28x28, int tensor, range: 0-255
            img_tensor_b = ((img_tensor*255)* np.clip(random.random(), 0.2 , 1)).int() # 1x28x28, int tensor, range: 0-255
            img_tensor = torch.cat([img_tensor_r, img_tensor_g, img_tensor_b], dim=0) # 3x28x28, int tensor, range: 0-255
            img_tensor = img_tensor / 255 # 3x28x28, float tensor, range: 0-255
        
        img_norm = (img_tensor*2)-1 # Cx28x28, float tensor, range: -1 to 1
        
        class_name = self.path_list[index].parent.stem
        class_label = self.class_to_idx[class_name]
        
        return img_norm, class_label
        
