import os
import sys
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch import nn, optim
import matplotlib.pyplot as plt
import torch
import tarfile
from vit import *

class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [os.path.join(directory, img) for img in os.listdir(directory) if img.endswith(('jpg', 'png', 'jpeg'))]
        self.labels = [self.extract_label(img) for img in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def extract_label(self, img_path):
        #print(img_path)
        name=img_path.split("_")
        if "attention" in img_path:
            label = 1
        else:
            label = 0 
        return label

def init():
    with tarfile.open("train.tar.gz", "r:gz") as tar:
        tar.extractall()
        tar.close()

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing images to match model input
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomPerspective(distortion_scale=0.2,p=0.5),
    # transforms.ColorJitter(brightness=.2,hue=.3),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(0,15))
    #Add more tran
])

# Loading your custom dataset
init()
train_dataset = CustomImageDataset(directory='./train', transform=transform)
print(len(train_dataset))
trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

def show_image(tensor_img):
    # Convert the tensor to a PIL image
    transform = transforms.ToPILImage()
    img = transform(tensor_img)

    # Plot the image using matplotlib
    plt.imshow(img)
    plt.axis('off')  # Turn off axis labels
    plt.show()
    
def check():
    for data in train_dataset:
        input, label = data
        show_image(input)
        break
        




check()