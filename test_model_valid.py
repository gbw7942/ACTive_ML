import torch
from torchvision import transforms
from PIL import Image
import random

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
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

def eval_attention_set():
    correct=0
    random_integers = [random.randint(0, 760) for _ in range(200)]
    for i in range(200):
        image_num=random_integers[i]+1
        result=check_pkl_accuracy("resnet_torch_model.pkl",f"./train/attention_{image_num}.jpg")
        if result==1:
            correct+=1
        else:
            print(image_num) 
    print(f"Test 200 images, correct number:{correct}, correct rate:{correct/200}")

def eval_phone_set():
    correct=0
    random_integers = [random.randint(0, 760) for _ in range(300)]
    for i in range(300):
        image_num=random_integers[i]+1
        result=check_pkl_accuracy("resnet_torch_model.pkl",f"./train/phone_{image_num}.jpg")
        if result==0:
            correct+=1
        else:
            print(image_num)
    print(f"Test 200 images, correct number:{correct}, correct rate:{correct/300}")


eval_attention_set()
eval_phone_set()

for i in range(24):
    result=check_pkl_accuracy("resnet_torch_model.pkl",f"{i+1}.jpg")
    if result==0:
        print("accurate")
    else:
        print("no") 