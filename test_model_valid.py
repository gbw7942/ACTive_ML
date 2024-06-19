import torch
from torchvision import transforms
from PIL import Image
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def check_pkl_validity(file_path):
    try:
        # Attempt to load the file
        data = torch.load(file_path)
        print("The file is valid.")
        return True
    except Exception as e:
        # Handle the exception if the file is not valid
        print(f"An error occurred: {e}")
        return False

# Example usage
# file_path = 'resnet_torch_model.pkl'
# is_valid = check_pkl_validity(file_path)

def check_pkl_accuracy(model_path,img_name):
    # Load the model using torch.load
    model = torch.load(model_path)
    
    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Set the model to evaluation mode
    model.eval()
    img=Image.open(img_name)
    img = transform(img)
    img = img.unsqueeze(0)
            
    # Forward pass through the model
    with torch.no_grad():
        output = model(img)
            
    # Get the predicted class (0 or 1) by finding the index of the max logit value
    _, predicted_class = torch.max(output, 1)
    
    return predicted_class.item()

def eval_set(attention,phone):
    # Make sure train folder is accessible through ./train
    correct_phone=0
    correct_attention=0
    wrong_phone=[]
    wrong_img=[]
    wrong_attention=[]
    random_attention_img=[random.randint(0, 760) for _ in range(attention)]
    random_phone_img=[random.randint(0, 760) for _ in range(phone)]
    for i in range(attention):
        attention_img_num=random_attention_img[i]+1
        attention_result=check_pkl_accuracy("resnet_torch_model.pkl",f"./train/attention_{attention_img_num}.jpg")
        if attention_result==1:
            correct_attention+=1
        else:
            wrong_attention.append(attention_img_num)
            wrong_img.append(f"./train/attention_{attention_img_num}.jpg")
    for j in range(phone):
        phone_img_num=random_phone_img[j]+1
        phone_result=check_pkl_accuracy("resnet_torch_model.pkl",f"./train/phone_{phone_img_num}.jpg")
        if phone_result==0:
            correct_phone+=1
        else:
            wrong_phone.append(phone_img_num)
            wrong_img.append(f"./train/phone_{phone_img_num}.jpg")

    print("---Summary---")
    print(f"Attention set tests {attention} images, correct number: {correct_attention}, correct rate: {correct_attention/attention}")
    print(f"Phone set tests {phone} images, correct number: {correct_phone}, correct rate: {correct_phone/phone}")
    if correct_attention!=attention:
        print(f"Wrong attention images:{wrong_attention}")
    if correct_phone!=phone:
        print(f"Wrong phone images:{wrong_phone}")
    if correct_phone+correct_attention!=phone+attention:
        num_images = len(wrong_img)
        cols = 5
        # Calculate number of rows required
        rows = num_images // cols + (num_images % cols != 0)

        # Create figure and axes
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()
        # Loop through image paths and display each image
        for i, img_path in enumerate(wrong_img):
            if os.path.exists(img_path):  # Check if image file exists
                img = mpimg.imread(img_path)
                axes[i].imshow(img)
                axes[i].axis('off')  # Hide the axis
            else:
                print(f"File not found: {img_path}")
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

eval_set(100,100)