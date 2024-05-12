
import os
import subprocess
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from starlette.responses import StreamingResponse, HTMLResponse
#from torchvision.tv_tensors import Image
from PIL import Image


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/test")
async def say_hello():

    return {"message": f"Hello "}


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


# Convert encoding data into 8-bit binary
# form using ASCII value of characters
def genData(data):
    newd = []

    for i in data:
        newd.append(format(ord(i), '08b'))
    return newd


# Pixels are modified according to the
# 8-bit binary data and finally returned
def modPix(pix, data):
    datalist = genData(data)
    lendata = len(datalist)
    imdata = iter(pix)

    for i in range(lendata):
        pix = [value for value in imdata.__next__()[:3] +
               imdata.__next__()[:3] +
               imdata.__next__()[:3]]

        for j in range(0, 8):
            if (datalist[i][j] == '0' and pix[j] % 2 != 0):
                pix[j] -= 1

            elif (datalist[i][j] == '1' and pix[j] % 2 == 0):
                if (pix[j] != 0):
                    pix[j] -= 1
                else:
                    pix[j] += 1

        if (i == lendata - 1):
            if (pix[-1] % 2 == 0):
                if (pix[-1] != 0):
                    pix[-1] -= 1
                else:
                    pix[-1] += 1
        else:
            if (pix[-1] % 2 != 0):
                pix[-1] -= 1

        pix = tuple(pix)
        yield pix[0:3]
        yield pix[3:6]
        yield pix[6:9]


def encode_enc(newimg, data):
    w = newimg.size[0]
    (x, y) = (0, 0)

    for pixel in modPix(newimg.getdata(), data):
        newimg.putpixel((x, y), pixel)
        if (x == w - 1):
            x = 0
            y += 1
        else:
            x += 1


# Encode data into image
@app.post("/encode/")
async def encode(file: UploadFile = File(...), data: str = Form(None)):
    try:
        image = Image.open(file.file, 'r')

        if data is None:
            raise ValueError('Data is empty')

        newimg = image.copy()
        print("data mã hóa ở đây: ", data)
        encode_enc(newimg, data)
        new_img_name = "test_eee.png"
        newimg.save(new_img_name, str(new_img_name.split(".")[1].upper()))
        return {"message": "Image encoded successfully!"}

    except Exception as e:
        return {"error": str(e)}


# Decode the data in the image
@app.post("/decode/")
async def decode(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file, 'r')

        data = ''
        imgdata = iter(image.getdata())

        while (True):
            pixels = [value for value in imgdata.__next__()[:3] +
                      imgdata.__next__()[:3] +
                      imgdata.__next__()[:3]]

            binstr = ''

            for i in pixels[:8]:
                if (i % 2 == 0):
                    binstr += '0'
                else:
                    binstr += '1'

            data += chr(int(binstr, 2))
            if (pixels[-1] % 2 != 0):
                return {"message": "Data decoded successfully!", "data": data}

    except Exception as e:
        return {"error": str(e)}

# import cv2
# import numpy as np
#
# # Đọc ảnh gốc và ảnh mới
# original_image = cv2.imread('test_eee.png')
# new_image = cv2.imread('test3.png')
#
# # Chuyển đổi sang grayscale
# original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
# new_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
#
# # Sử dụng template matching để tìm vị trí của ảnh gốc trong ảnh mới
# result = cv2.matchTemplate(new_gray, original_gray, cv2.TM_CCOEFF_NORMED)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#
# # Vị trí tốt nhất là max_loc
# top_left = max_loc
# bottom_right = (top_left[0] + original_gray.shape[1], top_left[1] + original_gray.shape[0])
#
# # Vẽ một hình chữ nhật quanh vùng tìm được
# cv2.rectangle(new_image, top_left, bottom_right, 255, 2)
#
# # Hiển thị ảnh
# cv2.imshow('Detected Area', new_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()