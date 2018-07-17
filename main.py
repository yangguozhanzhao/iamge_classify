import Tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import cv2
import os
import time
os.system ('sudo modprobe bcm2835-v4l2')

def show_cam():
	cap = cv2.VideoCapture(0) 
	cap.set(3,512)
	cap.set(4,512)
	cap.set(cv2.cv.CV_CAP_PROP_BRIGHTNESS,0.5)
	print cap.get(3),cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
	time.sleep(3)
	photo_number=0
	while 1:
		ret,frame=cap.read()
		frame=frame[100:400,100:400,:]
		time.sleep(0.01)
		cv2.imshow("capture", frame)
		#~ time.sleep(1)
		#~ print photo_number
		
		#~ photo_number+=1
		#~ filename="g_"+str(photo_number)+".jpeg"
		#~ path = "data/"+filename
		#~ cv2.imwrite(path,frame)
		
		#~ if photo_number>100:
			#~ break
		   
		if cv2.waitKey(1) & 0xFF == ord('q'):
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
