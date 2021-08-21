#!/usr/bin/env python
# coding: utf-8

# ## The EAST deep learning text detector
# ![image.png](attachment:image.png)
# <h4 style='text-align:center'> Hình 3: Cấu trúc của Mạng Fully Convolutional EAST (Zhou et al.). </h4>

# Thuật toán “EAST” là viết tắt của: **Efficient**  and  **Accurate Scene Text** detection pipeline.
# 
# Pipeline EAST có khả năng dự đoán các từ và dòng text ở các hướng tùy ý trên hình ảnh 720p và hơn nữa, có thể chạy ở tốc độ 13 FPS, theo các tác giả.
# 
# Để xây dựng và đào tạo một mô hình học sâu như vậy, phương pháp EAST sử dụng các hàm loss mới, được thiết kế cẩn thận.
# 

# In[1]:


import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import time


# **Quan trọng**: EAST yêu cầu kích thước hình ảnh input là bội số của 32, vì vậy ta phải điều chỉnh các giá trị - width và --height của mình, đảm bảo rằng chúng là bội số của 32 (ví dụ 32x32)

# In[2]:


# Load input image
image = cv2.imread('test.png')
copy_image = image.copy()
# grab the image dimensions
(H, W) = image.shape[:2]  

# set the new width and height and then determine the ratio in change for both the width and height
(newH, newW) = (320,320)                       # đảm bảo các kích thước là bội của 32
ratioW = float(W / newW)
ratioH = float(H / newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]
H,W


# Định nghĩa 2 output layer cho EAST model:

# In[3]:


outputLayers = [
    'feature_fusion/Conv_7/Sigmoid',      # output probabilities
    'feature_fusion/concat_3'         # derive the bounding box coordinates of text
]  


# In[4]:


# load pre-trained EAST model
east_model = cv2.dnn.readNet('E:/My Project/Real-time OCR & Text Detection with Tensorflow, OpenCV and Tesseract/pretrained model/frozen_east_text_detection.pb')

# construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0, (W,H), 
                            (123.68, 116.78, 103.94), swapRB=True, crop=False)


# In[5]:


# then perform a forward pass of the model to obtain the two output layer sets
east_model.setInput(blob)

start = time.time()
proba, coordinate = east_model.forward(outputLayers)       # xuất ra 2 feature map thể hiện xác suất dự đoán và tọa độ bounding box
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))


# Sau khi đã có các output từ model EAST, trích xuất các thông tin về tọa độ các bounding box, và xác suất tin cậy tương ứng của chúng:

# In[6]:


# Định nghĩa hàm prediction_featuring để tiện sử dụng sau này
def prediction_featuring(proba, coordinate, min_confidence=0.5):
    # grab the number of rows and columns from the probabilities volume 
    n_rows, n_cols = proba.shape[2:4]
    # initialize our set of bounding box rectangles and corresponding confidence scores
    rects = []      # Lưu trữ hộp bounding box (x, y) - tọa độ cho các vùng text
    confidences = []    # Lưu trữ xác suất được liên kết với mỗi box trong rects

    # loop over the number of rows
    for y in range(n_rows):
        # Trích xuất các proba (xác suất)
        prob = proba[0,0,y]
        # trích xuất các dữ liệu tọa độ bounding box tiềm năng
        xData0 = coordinate[0, 0, y]
        xData1 = coordinate[0, 1, y]
        xData2 = coordinate[0, 2, y]
        xData3 = coordinate[0, 3, y]
        anglesData = coordinate[0, 4, y]

        # loop over the number of columns
        for x in range(n_cols):
            # nếu xs tin cậy nhỏ hơn mức tối thiểu thì bỏ qua box này
            if prob[x] < min_confidence:
                continue
            # tính toán hệ số offset vì các feature map của ta sẽ nhỏ hơn 4 lần so với hình ảnh đầu vào
            offsetX, offsetY = x*4.0, y*4.0

            # trích xuất góc quay cho dự đoán và sau đó tính sin và cosine
            angle = anglesData[x]
            sin = np.sin(angle)
            cos = np.cos(angle)

            # Tính toán heigth và width của bouding box (theo định nghĩa từ model)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # tính cả tọa độ bắt đầu và kết thúc (x, y) cho bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # Thêm các giá trị tọa độ vừa tính đc và xác suất tin cậy vào các list
            rects.append((startX, startY, endX, endY))
            confidences.append(prob[x])
        
    return (rects, confidences)


# Bước cuối cùng là áp dụng **Non-max Suppression** (triệt tiêu phi cực đại) cho các bounding box đang có để loại bỏ các box chồng nhau có xác suất tin cậy thấp. Cuối cùng là hiển thị các kết quả nhận diện Text:

# In[ ]:


# apply non-maxima suppression to suppress weak, overlapping bounding boxes
rects, confidences = prediction_featuring(proba, coordinate, min_confidence=0.6)
boxes = non_max_suppression(np.array(rects), probs=confidences)

# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
    # resize lại tỷ lệ tọa độ bounding box dựa trên ratio
    startX = int(startX * ratioW)
    startY = int(startY * ratioH)
    endX = int(endX * ratioW)
    endY = int(endY * ratioH)
    
    # draw the bounding box on the image
    cv2.rectangle(copy_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    
# show the output image
cv2.imshow("Text Detection", copy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




