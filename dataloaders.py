from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, type_ ='classes' ,transform=None, target_transform=None, standarization=False):
        self.particles = []
        self.defocus = []
        self.img_dir = img_dir
        self.type = type_
        self.transform = transform
        self.target_transform = target_transform

        # List all files in the directory (assuming they are images)
        self.img_names = os.listdir(img_dir)
        #####################################################################################
        #defocus_value = 230  #BASE
        #####################################################################################
        for imname in self.img_names:
            ###############################
            if(type_=='sequential'):
                defocus_value = 230 + 2*(int(imname.split('_')[1])-1)
            if(type_=='classes'):
                defocus_class = (int(imname.split('_')[1])-1)
            ###############################

            img_path = os.path.join(self.img_dir, imname)  # get specific image path
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
                #image = (np.array(image) / 255)[None, :, :]  # Normalize to [0, 1]

            image = (image - image.mean())/ image.std()  # added standardization per image!!!                     !!!
            
            self.particles.append(np.array(image))
            if(type_=='classes'):
                self.defocus.append(defocus_class)
            if(type_=='sequential'):
                self.defocus.append(defocus_value)
        
        self.particles = np.array(self.particles)
        
        # Standardization if specified
        if standarization:
            self.particles = (self.particles - self.particles.mean()) / self.particles.std()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        defocus_tensor = torch.tensor(self.defocus[idx], dtype=torch.long)
        return self.particles[idx],defocus_tensor

def get_dataloader(paths_to_data, batch_size, standarization=False,type_='classes',  train_ratio=0.8):
    """Loads datasets from the provided list of paths.

    paths_to_data : list
        A list of paths to dataset directories.
    batch_size : int
        The batch size for the dataloader.
    """
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor()
    ])
    transform = None

    '''
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  
])    
'''
    # Use the first path from the list of data paths (can be adjusted if multiple datasets are needed)
    selected_path = paths_to_data[0]

    # Get dataset
    particle_dset = CustomImageDataset(
        img_dir=selected_path,
        type_=type_,
        transform=transform,
        standarization=standarization
        
    )
    # Calculate the sizes for training and test splits
    total_size = len(particle_dset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size

    # Split the dataset into training and test sets
    train_dataset, test_dataset = random_split(particle_dset, [train_size, test_size])

    # Create DataLoaders for both train and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, test_loader
