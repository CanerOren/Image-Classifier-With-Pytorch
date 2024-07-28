# D:/Anaconda/python.exe d:/Yazılım/image-clasiffier/gradio_infer.py
import sys

sys.path.append("D:/Anaconda/envs/classifier/Lib/site-packages")
import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gradio as gr
import torch
import torchvision.transforms as transforms

from model import ExampleModel

model_path="./model/animal_7.pth"
labels=['bird','cat','dog','horse']
num_classes=len(labels)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)

model= ExampleModel(num_classes=num_classes)
model.load_state_dict(torch.load(model_path,map_location=device))
model.to(device).eval()

def predict(image):
    img_tensor=transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(img_tensor)
    prediction = torch.nn.functional.softmax(prediction,dim=1).squeeze()
    confidences={labels[i]:float(prediction[i]) for i in range(num_classes)}
    return confidences

gr.Interface(fn=predict , inputs=gr.Image(type='pil'),outputs=gr.Label(num_top_classes=num_classes)).launch()
