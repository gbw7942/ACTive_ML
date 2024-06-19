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
import resnet

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
trainloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

def train():
    EPOCH_NUM = 6
    LEARN_R = 0.0001  # or lr=0.001
    model = resnet18(weights=None) # resnet.resnet18() # ViT()
    # model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: focus or not
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARN_R, momentum=0.9)
    
    epoch_losses = []
    
    for epoch in range(EPOCH_NUM):
        loss_avg = 0
        num_img = 0
        for data in trainloader:
            inputs, labels = data
            model.train()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_avg += loss.item() * len(labels)
            num_img += len(labels)
            sys.stdout.flush()
        avg_loss = loss_avg / num_img
        epoch_losses.append(avg_loss)
        print('\r', end="")
        print('正在训练：{}/{}轮，学习率为：{:.10f}，平均Loss：{:.2f}'
              .format(epoch + 1, EPOCH_NUM, optimizer.state_dict()['param_groups'][0]['lr'], avg_loss))
    model_state_dict = model.state_dict()
    torch.save(model, 'resnet_torch_model.pkl')
    print("Model saved to model.pkl")
    
    # Plotting the loss
    plt.figure()
    plt.plot(range(EPOCH_NUM), epoch_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()

# Train the model
train()