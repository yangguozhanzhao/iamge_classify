import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import cv2
import os
import time
os.system ('sudo modprobe bcm2835-v4l2')

def show_cam():
    cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH,240)  
   # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,320)
    photo_number=0
    while 1:
        ret,frame=cap.read()
        cv2.imshow("capture", frame)
        
        if cv2.waitKey(500) & 0xFF == ord('a'):
            photo_number+=1
            filename="a_"+str(photo_number)+".jpeg"
            path = "data/"+filename
            time.sleep(0.5)
            cv2.imwrite(path,frame)

        elif cv2.waitKey(600) & 0xFF == ord('b'):
            photo_number+=1
            filename="b_E"+str(photo_number)+".jpeg"
            path = "data/"+filename
            time.sleep(0.3)
            cv2.imwrite(path,frame)

        elif cv2.waitKey(650) & 0xFF == ord('c'):
            photo_number+=1
            filename="c_"+str(photo_number)+".jpeg"
            path = "data/"+filename	
            cv2.imwrite(path,frame)
            
        elif cv2.waitKey(700) & 0xFF == ord('q'):
            break
            cap.release()
            cv2.destroyAllWindows()

def take_photo(category):
    photo_number+=1
    filename=category+'_'+str(self.photo_number)+".jpeg"
    path="data/"+filename
    ret,frame = self.cap.read()
    cv2.imwrite(path, frame)
    print(self.photo_number)

    def train(self):
        pass
    def predict(self):
        pass


if __name__ == "__main__":
    show_cam()
