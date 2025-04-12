from ultralytics import YOLO
import numpy
from pathlib import Path
import time
import cv2

path = Path(__file__).parent
cam = cv2.VideoCapture(0)
#image = cv2.imread("scirock.jpg")
model_path = path / "best.pt"
model = YOLO(model_path)
cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)
def knb(p1,p2):
    if (p1=="Rock" and p2=="Scissors") or (p1=="Paper" and p2=="Rock") or (p1=="Scissors" and p2=="Paper"):
        return 1
    elif p1==p2:
        return 0
    else:
        return 2
state = "idle" #wait, result
prev_time = 0
curr_time = 0
winner=-1
player1_hand=""
player2_hand=""
while cam.isOpened():
    print(state)
    rat, frame = cam.read()
    #frame = image.copy()
    results = model(frame)
    #ann_frame = results[0].plot()
    if state=="wait" and (curr_time - prev_time)>5:
        prev_time=time.time()
        state="fight"
    elif state=="wait":
        curr_time=time.time()
        show_time=5-int(curr_time - prev_time)
        cv2.putText(frame, f"Start in {show_time}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0),2)
    if state=="show_winner" and (curr_time - prev_time)>5:
        winner==-1
        state="idle"
    elif state=="show_winner":
        curr_time=time.time()
        winner_text=""
        if winner==0:
            winner_text="Draw"
        else:
            winner_text=f"Player {winner} is winner"
        cv2.putText(frame, winner_text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0),2)
    if state=="fight" and (curr_time - prev_time)>5:
        state="idle"
    elif state=="fight":
        curr_time=time.time()
    result=results[0]
    if len(result.boxes.xyxy)==2 and state != "wait":
        members=[]
        for i in range(len(result.boxes.xyxy)):
            x1,y1,x2,y2 = result.boxes.xyxy[i].numpy().astype("int")
            cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)
            cv2.putText(frame, result.names[result.boxes.cls[i].item()], (x1+10, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0),2)
            members+=[result.names[result.boxes.cls[i].item()]]
        player1_hand, player2_hand = members
        if player1_hand==player2_hand and player2_hand=="Rock" and state=="idle":
            state="wait"
            prev_time=time.time()
            curr_time=time.time()
        if state=="fight":
            winner = knb(player1_hand,player2_hand)
            state="show_winner"
            prev_time=time.time()
            curr_time=time.time()
    cv2.imshow("YOLO",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()



