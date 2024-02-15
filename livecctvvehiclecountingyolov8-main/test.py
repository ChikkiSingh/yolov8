import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
from vidgear.gears import CamGear
from tracker import*
model=YOLO('yolov8s.pt')

stream = CamGear(source='https://www.youtube.com/watch?v=ByED80IKdIU', stream_mode = True, logging=True).start() # YouTube Video URL as input
# import cv2

# def main():
#     cap = cv2.VideoCapture(1)
#     while True:
#         ret, frame = cap.read()
       
        
#######
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)




my_file = open("F:\livecctvvehiclecountingyolov8-main\livecctvvehiclecountingyolov8-main\coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)
count=0
tracker =Tracker()
#area1=[(395,447),(852,382)]
#area2=[(413,454),(857,390)]
area1=[(397,450),(877,377),(864,395),(399,462)]
area2=[(401,475),(892,397),(995,412),(430,488)]

downcar={}
downcarcounter=[]
upcar={}
upcarcounter=[]
#area1=[(752,263),(414,384),(437,396),(772,272)]


while True:    
    frame = stream.read()   
    count += 1
    if count % 2 != 0:
        continue


    frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame)
    #results1=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
   # a=results1[0].boxes.data
    # c=results2[0].boxes.data
    # d=results3[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
            list.append([x1,y1,x2,y2])
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        x3,y3,x4,y4,id1=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        result=cv2.pointPolygonTest(np.array(area1,np.int32),((cx,cy)),False)
        if result>=0:
            downcar[id1]=(cx,cy)
        if id1 in downcar:
            result1=cv2.pointPolygonTest(np.array(area2,np.int32),((cx,cy)),False)
            if result1>=0:
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2) 
                cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
                if  downcarcounter.count(id1)==0:
                    downcarcounter.append(id1)
            
###################################################################################################
        results2=cv2.pointPolygonTest(np.array(area2,np.int32),((cx,cy)),False)
        if results2>=0:
            upcar[id1]=(cx,cy)
        if id1 in upcar:
            results3=cv2.pointPolygonTest(np.array(area1,np.int32),((cx,cy)),False)
            if results3>=0:
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2) 
                cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
                if upcarcounter.count(id1)==0:
                   upcarcounter.append(id1)
   #########

    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,255,0),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,0),2)
    card=len(downcarcounter)
    caru=len(upcarcounter)
    cvzone.putTextRect(frame,f'downsideLane:{card}',(750,60),1,1)
    cvzone.putTextRect(frame,f'upsideLane:{caru}',(50,60),1,1)

    cv2.imshow("RGB", frame)
    

    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()



