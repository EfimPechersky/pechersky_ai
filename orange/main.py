from ultralytics import YOLO
import numpy
from pathlib import Path
import time
import cv2
import matplotlib.pyplot as plt
from skimage import draw
path = Path(__file__).parent
cam = cv2.VideoCapture(0)
#image = cv2.imread("scirock.jpg")
model_path = path / "facial_best.pt"
image = cv2.imread("photo.png")
orig_oranges=cv2.imread("oranges.png")

hsv_orange = cv2.cvtColor(orig_oranges, cv2.COLOR_BGR2HSV)
lower=numpy.array([5,240,200])
upper=numpy.array([15,255,255])

mask=cv2.inRange(hsv_orange,lower,upper)
kernel=numpy.ones((7,7))
mask=cv2.erode(mask, kernel)
mask=cv2.dilate(mask, kernel,iterations=4)
contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sorted_contours=sorted(contours, key=cv2.contourArea)
m=cv2.moments(sorted_contours[-1])
cx = int(m["m10"] / m["m00"])
cy = int(m["m01"] / m["m00"])
bbox = cv2.boundingRect(sorted_contours[-1])
model = YOLO(model_path)
while cam.isOpened():
    ret,frame=cam.read()
    result = model(frame)[0]
    masks=result.masks
    if masks==None:
        continue
    ann = result.plot()
    global_mask = masks[0].data.numpy()[0,:,:]
    for mask in masks[1:]:
        global_mask +=mask.data.numpy()[0,:,:]
    global_mask = cv2.resize(global_mask, (image.shape[1],image.shape[0]))
    rr, cc = draw.disk((5,5), 5)
    struct = numpy.zeros((11,11),numpy.uint8)
    struct[rr,cc]=1
    global_mask=cv2.dilate(global_mask, struct, iterations=1)

    res_mask = numpy.zeros_like(image)
    res_mask[:,:,0]=global_mask
    res_mask[:,:,1]=global_mask
    res_mask[:,:,2]=global_mask
    parts=(image*res_mask).astype(numpy.uint8)
    pos = numpy.where(res_mask>0)
    margin=30
    min_y, min_x = numpy.min(pos[0])-margin, numpy.min(pos[1])-margin
    max_y, max_x = numpy.max(pos[0])+margin, numpy.max(pos[1])+margin
    res_mask=res_mask[min_y:max_y, min_x:max_x]
    parts=parts[min_y:max_y, min_x:max_x]
    if 0 in parts.shape:
        continue
    resized_parts = cv2.resize(parts, (bbox[2],bbox[3]))
    resized_mask = cv2.resize(res_mask, (bbox[2],bbox[3]))*255
    oranges=orig_oranges.copy()
    x,y,w,h=bbox
    roi = oranges[y:y+h,x:x+w]
    bg = cv2.bitwise_and(roi,roi,mask=cv2.bitwise_not(resized_mask[:,:,0]))
    comb = cv2.add(bg, resized_parts)
    oranges[y:y+h, x:x+w]=comb
    cv2.imshow("Mask", oranges)
    key = cv2.waitKey(1)
    if key==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()



