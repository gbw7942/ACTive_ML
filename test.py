import torch
from torchvision import transforms
from PIL import Image
import cv2
import threading
import time


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing images to match model input
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image)#.unsqueeze(0)
    return image

def test(model_path, processed_img):
    model = torch.load(model_path)
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():
        prediction = model(processed_img)
        print(prediction)

def capture_frame(): #获取图片
    video_url = "https://iusopen.ezvizlife.com/v3/openlive/L37100666_1_1.m3u8?expire=1778165089&id=711710851078774784&c=82f56de598&t=2096574bdff8f10b8b8e944315f6b97f935e8887c6bddbd73ecdcf46aac1e09d&ev=100"
    # Open the video stream
    cap = cv2.VideoCapture(video_url)
    if cap.isOpened():
        ret, frame = cap.read() # frame - 预处理的图片     
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 将BGR格式转换为RGB格式
        image = Image.fromarray(frame_rgb) # 将NumPy数组转换为PIL图像
        # [optional] TODO - 定义一个时间frame去抽帧
        processed_image = preprocess_image(image).unsqueeze(0) # 处理图片
        test(model_path='resnet_torch_model.pkl', processed_img=processed_image) # query model (test)


threading.Timer(0.5, capture_frame).start()
