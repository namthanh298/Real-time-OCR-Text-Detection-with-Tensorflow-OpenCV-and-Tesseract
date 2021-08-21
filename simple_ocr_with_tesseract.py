#!/usr/bin/env python
# coding: utf-8


# import the necessary packages
from PIL import Image
import pytesseract
import cv2
import os
import numpy as np
import re
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r'C:/.../tesseract.exe'           # trỏ đến đường dẫn của core file của thư viện Tesseract OCR: tesseract.exe




# load the example image and convert it to grayscale
image = cv2.imread('test.jpg')

""" Một số phép tiền xử lý ảnh trước khi đưa qua Tesseract nhận diện """
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# check to see if we should apply thresholding to preprocess the image
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# make a check to see if median blurring should be done to remove noise
gray = cv2.medianBlur(gray, 3)
# write the grayscale image to disk as a temporary file so we can apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, image)
plt.imshow(image)


# Cuối cùng chúng ta có thể áp dụng OCR cho hình ảnh của mình bằng cách sử dụng Tesseract Python:


# load the image as a PIL/Pillow image, apply OCR, and then delete the temporary file
text = pytesseract.image_to_string(Image.open(filename), lang='eng+vie')
os.remove(filename)
print(text)


# # Thực hiện Text Localization, Text Detection và OCR với Tesseract


# tải hình ảnh đầu vào, chuyển đổi từ kênh BGR sang RGB,
# sử dụng Tesseract để localize từng vùng text trong hình ảnh đầu vào
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
objects = pytesseract.image_to_data(rgb, lang='eng+vie', output_type=pytesseract.Output.DICT)
objects.keys()          # In ra các key chính của objects



# loop over each of the individual text localizations
for i in range(0, len(objects["text"])):
    # extract the bounding box coordinates of the text region from the current result
    x = objects["left"][i]
    y = objects["top"][i]
    w = objects["width"][i]
    h = objects["height"][i]
    # extract the OCR text itself along with the confidence of the text localization
    text = objects["text"][i]
    conf = float(objects["conf"][i])
    
    if text.isspace(): continue 
    
    if conf > 85:
        # display the confidence and text to our terminal
        print("Confidence: {}".format(conf))
        print("Text: {}".format(text))
        print("")
        # tách các đoạn text hoặc kí tự không phải ASCII 
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        # sử dụng OpenCV, sau đó vẽ một bounding box cho text
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
        #            color=(0, 0, 255), fontScale=1, thickness=2)
        
cv2.imshow("Image with detected text", image)
cv2.waitKey(0)
cv2.destroyAllWindows()




