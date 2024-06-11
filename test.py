import torch
from torchvision import transforms
from PIL import Image
import cv2
import threading
import time
import os


def capture_frame(): #获取图片
    video_url = "https://iusopen.ezvizlife.com/v3/openlive/L37100666_1_1.m3u8?expire=1778165089&id=711710851078774784&c=82f56de598&t=2096574bdff8f10b8b8e944315f6b97f935e8887c6bddbd73ecdcf46aac1e09d&ev=100"
    # Open the video stream
    cap = cv2.VideoCapture(video_url)
    if cap.isOpened():
        frame = cap.read() # frame - 预处理的图片     
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 将BGR格式转换为RGB格式
        image = Image.fromarray(frame_rgb) # 将NumPy数组转换为PIL图像
        # [optional] TODO - 定义一个时间frame去抽帧
        result=test_imgs(model_path='resnet_torch_model.pkl', img=image) # query model (test)
        print(result)


# threading.Timer(0.5, capture_frame).start()

def test_imgs(model_path,img):
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
    img = transform(img)
    img = img.unsqueeze(0)
            
    # Forward pass through the model
    with torch.no_grad():
        output = model(img)
            
    # Get the predicted class (0 or 1) by finding the index of the max logit value
    _, predicted_class = torch.max(output, 1)
    
    return predicted_class.item()

def main():
    capture_frame()
    time.sleep(0.5)
    main()