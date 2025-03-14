import os
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random
import torch

def find_classes(directory):
    directory = Path(directory)
    
    class_names = sorted(entry.name for entry in os.scandir(directory))
    if not class_names:
        raise FileNotFoundError(f"[INFO] No valid class names can be found in {directory}.")
    
    class_to_idx = {}
    for idx, name in enumerate(class_names):
        class_to_idx[name] = idx
    
    return class_names, class_to_idx


class custom_dataset(Dataset):
    
    def __init__(self, directory, config):
        directory = Path(directory)
        self.path_list = list(directory.glob("*/*.png"))
        self.config = config
        self.classes, self.class_to_idx = find_classes(directory)
    
    def load_image(self, index):
        img = Image.open(self.path_list[index])
        return img
    
    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, index):
        
        img = self.load_image(index)
        
        simple_transform = transforms.ToTensor()
        img_tensor = simple_transform(img) # float tensor, 0-1, CxHxW, C=1
        
        if self.config['input_channels'] == 3:
            img_tensor_r = (img_tensor*255).int()*torch.clip(torch.rand((1,)), 0.2, 1) # int tensor, 0-255, CxHxW, C=1
            img_tensor_g = (img_tensor*255).int()*torch.clip(torch.rand((1,)), 0.2, 1) # int tensor, 0-255, CxHxW, C=1
            img_tensor_b = (img_tensor*255).int()*torch.clip(torch.rand((1,)), 0.2, 1) # int tensor, 0-255, CxHxW, C=1
            img_tensor = torch.cat([img_tensor_r, img_tensor_g, img_tensor_b], dim=0) # int tensor, 0-255, CxHxW, C=3, rgb
            img_tensor = img_tensor / 255 # float tensor, 0-1, CxHxW, C=3, rgb
        
        img_norm = (img_tensor*2) - 1 # float tensor, -1 to 1
        
        class_name = self.path_list[index].parent.stem
        class_label = self.class_to_idx[class_name]
        
        return img_norm, class_label
        
        