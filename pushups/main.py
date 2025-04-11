from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy
from pathlib import Path
import time
import cv2
def dist(a,b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

def angle(a,b,c):
    d=numpy.rad2deg(numpy.arctan2(c[1]-b[1], c[0] - b[0]))
    e=numpy.rad2deg(numpy.arctan2(a[1]-b[1], a[0] - b[0]))
    ans=d-e
    ans = ans+360 if ans<0 else ans
    return 360-ans

def process(image, keypoints):
    nose_seen = keypoints[0][0]>0 and keypoints[0][1]>0
    left_ear_seen = keypoints[3][0]>0 and keypoints[3][1]>0
    right_ear_seen =keypoints[4][0]>0 and keypoints[4][1]>0
    left_shoulder = keypoints[5]
    right_shoulder =keypoints[6]
    left_elbow =keypoints[7]
    right_elbow =keypoints[8]
    left_wrist =keypoints[9]
    right_wrist =keypoints[10]
    left_hip =keypoints[11]
    right_hip =keypoints[12]
    left_knee =keypoints[13]
    right_knee =keypoints[14]
    left_ankle =keypoints[15]
    right_ankle =keypoints[16]
    angle_arm=0
    angle_body=0
    try:
        if left_ear_seen and not right_ear_seen:
            angle_arm = angle(left_shoulder, left_elbow, left_wrist)
            angle_body = angle(left_shoulder, left_hip, (left_shoulder[0], left_hip[1]))
        else:
            angle_arm = angle(right_shoulder, right_elbow, right_wrist)
            angle_body = angle(right_shoulder, right_hip, (right_shoulder[0], right_hip[1]))
        x, y = int(left_elbow[0])+10, int(left_elbow[1])+10
        cv2.putText(frame,f"angle:{angle_arm}",(x,y), cv2.FONT_HERSHEY_PLAIN, 1.5,(25,255,25),1)
        if angle_body<330 and angle_body>20:
            angle_arm=0
        return angle_arm
    except ZeroDivisionError:
        pass
path = Path(__file__).parent
model_path = path / "yolo11n-pose.pt"
model = YOLO(model_path)
#cam = cv2.VideoCapture('rtsp://10.76.55.112:8000/h264.sdp')
cam = cv2.VideoCapture(0)
cv2.namedWindow("ORIG", cv2.WINDOW_NORMAL)
last_time = time.time()
count=0
is_sitting=False
writer = cv2.VideoWriter("out.avi", cv2.VideoWriter_fourcc(*"XVID"),10,(640,400))
while cam.isOpened():
    ret, frame = cam.read()
    curr_time=time.time()
    cv2.putText(frame,f"{count}",(10,20), cv2.FONT_HERSHEY_PLAIN, 1.5,(25,255,25),1)
    cv2.imshow("ORIG", frame)
    results = model(frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
    if not results:
        continue
    result=results[0]
    keypoints = result.keypoints.xy.tolist()
    if not keypoints:
        continue
    keypoints=keypoints[0]
    if not keypoints:
        continue
    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
    annotated = annotator.result()
    ang = process(annotated, keypoints)
    if ang>0:
        if not is_sitting and ang<=80:
            count+=1
            is_sitting=True
            last_time = curr_time
        elif is_sitting and ang>140:
            is_sitting=False
            last_time = curr_time
    if (curr_time-last_time)>90:
        count=0
    annotated = cv2.resize(annotated, (640,400), interpolation= cv2.INTER_LINEAR)
    writer.write(annotated)
    cv2.imshow("YOLO", annotated)
    
    
cam.release()
writer.release()
cv2.destroyAllWindows()




