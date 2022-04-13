import cv2
import numpy as np

vid = cv2.VideoCapture("0,0,0 - 05-April-2022 10:28:33(1).mp4")
count = 1
while(vid.isOpened()):
    ret, frame = vid.read()
    if ret == True:
        cv2.imshow('frame', frame)
        
        filename = str(count) + ".jpg"
        if cv2.waitKey(1) & 0xff == ord("s"):
            cv2.imwrite(filename, frame)
            count+=1
            
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    else:
        break

vid.release()
cv2.destroyAllWindows()