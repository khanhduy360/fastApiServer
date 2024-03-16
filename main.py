import base64
import os
import subprocess
from io import BytesIO

import torch
import torchvision.transforms as transforms

from fastapi import FastAPI, HTTPException, UploadFile, File
from starlette.responses import StreamingResponse, HTMLResponse
from torchvision.tv_tensors import Image

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/test")
async def say_hello():
    # Load the model
    model = torch.load('logo_classifier.pth')
    model.eval()

    # Define the transformation for input images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Function to predict the label of an input image
    def predict_image(image_path, model, transform):
        # Open the image file
        image = Image.open(image_path)
        # Apply transformations
        image = transform(image).unsqueeze(0)  # Add batch dimension
        # Perform inference
        with torch.no_grad():
            output = model(image)
        # Get predicted class index
        _, predicted = torch.max(output, 1)
        # Map the index to class label
        class_index = predicted.item()
        # Return the class label
        return class_index

    # Path to the image you want to classify
    image_path = 'OIP.jpg'

    # Predict the label of the image
    predicted_label_index = predict_image(image_path, model, transform)

    # Print the predicted label
    class_names = ['Leisure', 'Institution', 'Cosmetic', 'Necessities', 'Medical', 'Electronic', 'Clothes',
                   'Transportation', 'Food', 'Accessories']
    predicted_label = class_names[predicted_label_index]
    print('Predicted label:', predicted_label)

    return {"message": f"Hello {predicted_label}"}


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):

    contents = await file.read()
    # Return an HTML response displaying the image
    # Lưu tạm thời nội dung ảnh vào một tệp
    with open("uploaded_image.jpg", "wb") as f:
        f.write(contents)
        # Đường dẫn tuyệt đối đến tệp ảnh
    image_path = os.path.abspath("uploaded_image.jpg")

    # Mở tệp ảnh bằng trình xem ảnh mặc định trên hệ thống Windows
    try:
        subprocess.run(["start", image_path], shell=True)
    except FileNotFoundError:
        print("Cannot find default image viewer.")
        # Xử lý lỗi hoặc thử sử dụng các lệnh mở tệp ảnh trên các hệ điều hành khác

    # Trả về thông báo thành công
    return {"message": "Image opened successfully"}