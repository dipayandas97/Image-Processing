import numpy as np
import cv2
from scipy.spatial.distance import euclidean

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
img = np.zeros((480,640))
prev_frame = prev_frame.astype('float64')

while(True):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    img = np.subtract(frame.astype('float64'), prev_frame.astype('float64')) / 256
    
    #print(frame[100][100],prev_frame[100][100], img[100][100]) 
    cv2.imshow('frame',img)
    prev_frame = frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
