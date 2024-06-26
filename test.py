import torch
from torchvision import transforms
from PIL import Image
import cv2
import time
from flask import Flask, jsonify

app = Flask(__name__)

class LeftTwoThirdsCrop(object):
    def __call__(self, img):
        # Get the dimensions of the image
        width, height = img.size
        
        # Calculate the width of the left 2/3 part
        new_width = int(width * 2 / 3)
        
        # Crop the image
        img_cropped = img.crop((0, 0, new_width, height))
        
        return img_cropped
    
def get_result(): #获取图片
    print("Getiing image")
    video_url = "https://iusopen.ezvizlife.com/v3/openlive/L37100666_1_1.m3u8?expire=1778165089&id=711710851078774784&c=82f56de598&t=2096574bdff8f10b8b8e944315f6b97f935e8887c6bddbd73ecdcf46aac1e09d&ev=100"
    # Open the video stream
    cap = cv2.VideoCapture(video_url)
    if cap.isOpened():
        ret,frame = cap.read() # frame - 预处理的图片
        if not ret:
            print("cannot read the frame")
            return -1
        else:
            print("retireved image success")    
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 将BGR格式转换为RGB格式
            image = Image.fromarray(frame_rgb) # 将NumPy数组转换为PIL图像
            # [optional] TODO - 定义一个时间frame去抽帧
            result=test_imgs(model_path='resnet_unofficial_model.pkl', img=image) # query model (test)
            print(result)
            return result
    else:
        print("camera not opened")
        return -2


def test_imgs(model_path,img):
    print("Loading model")
    # Load the model using torch.load
    model = torch.load(model_path)
    
    # Define the image transformations
    transform = transforms.Compose([
        LeftTwoThirdsCrop(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
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
    print("Predicting")
    
    return predicted_class.item()

# @app.route('/get_integer', methods=['GET'])
# def get_integer():
#     result = get_result()
#     data = {"value": result}
#     return jsonify(data)

# if __name__ == '__main__':
#     app.run(host="10.0.0.66",port=5000)

def start():
    get_result()
    time.sleep(1)
    start()
start()